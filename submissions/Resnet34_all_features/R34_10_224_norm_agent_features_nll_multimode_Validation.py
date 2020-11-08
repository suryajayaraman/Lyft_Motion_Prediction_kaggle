#!/usr/bin/env python
# coding: utf-8

# ## Library imports

# common imports
import os
import numpy as np
from tqdm import tqdm
import random
import time
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from tempfile import gettempdir
import matplotlib.pyplot as plt
from typing import Dict

# torch imports
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.models.resnet import resnet50, resnet18, resnet34, resnet101
import torch.nn.functional as F

# l5kit imports
import l5kit
from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import AgentDataset, EgoDataset
from l5kit.rasterization import build_rasterizer
from l5kit.evaluation import write_pred_csv, compute_metrics_csv, read_gt_csv, create_chopped_dataset
from l5kit.evaluation.chop_dataset import MIN_FUTURE_STEPS
from l5kit.evaluation.metrics import neg_multi_log_likelihood, time_displace
from l5kit.geometry import transform_points
from l5kit.visualization import PREDICTED_POINTS_COLOR, TARGET_POINTS_COLOR, draw_trajectory


print('l5kit.__version__' , l5kit.__version__)


print('torch.cuda.is_available()', torch.cuda.is_available())


def find_no_of_trainable_params(model):
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #print(total_trainable_params)
    return total_trainable_params


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


# ## Configs


# --- Lyft configs ---
cfg = {
    'format_version': 4,
    'data_path': "../../lyft-motion-prediction-autonomous-vehicles/",
    'model_params': {
        'model_architecture': 'resnet34',
        'history_num_frames': 10,
        'history_step_size': 1,
        'history_delta_time': 0.1,
        'future_num_frames': 50,
        'future_step_size': 1,
        'future_delta_time': 0.1,
        'model_name': "R34_pvt_10_224_norm_agent_features_nll_multimode",
        'lr': 1e-3 * 0.4,
        'weight_path': "R34_pvt_10_224_norm_agent_features_nll_multimode_2176k.pth",
        'train': True,
        'predict': False
    },

    'raster_params': {
        'raster_size': [224, 224],
        'pixel_size': [0.5, 0.5],
        'ego_center': [0.25, 0.5],
        'map_type': 'py_semantic',
        'satellite_map_key': 'aerial_map/aerial_map.png',
        'semantic_map_key': 'semantic_map/semantic_map.pb',
        'dataset_meta_key': 'meta.json',
        'filter_agents_threshold': 0.5,
        'disable_traffic_light_faces' : False
    },
    
    'train_data_loader': {
        'key': 'scenes/train.zarr',
        'batch_size': 32,
        'shuffle': True,
        'num_workers': 4
    },

    'val_data_loader': {
        'key': 'scenes/validate.zarr',
        'batch_size': 16,
        'shuffle': True,
        'num_workers': 4
    },

    'test_data_loader': {
        'key': 'scenes/test.zarr',
        'batch_size': 32,
        'shuffle': False,
        'num_workers': 4
    },

    'train_params': {
        'train_start_index' : 7501,
        'max_num_steps': 7505,
        'checkpoint_every_n_steps': 250,
    }
}



NUMBER_OF_HISTORY_FRAMES = cfg['model_params']['history_num_frames'] + 1
RASTER_IMG_SIZE = cfg['raster_params']['raster_size'][0]
NUM_MODES = 3
NUMBER_OF_FUTURE_FRAMES = cfg['model_params']['future_num_frames']
TRAIN_BATCH_SIZE = cfg['train_data_loader']['batch_size'] 
### TRAIN FROM WHERE LEFT OFF, CHANGE THE STARTING INDICES VARIABLE ACCORDINGLY
TRAIN_START_INDICES = cfg['train_params']['train_start_index']
EXTENT_RANGE = 5.0 


# ## Load the data

# set env variable for data
DIR_INPUT = cfg["data_path"]
os.environ["L5KIT_DATA_FOLDER"] = DIR_INPUT
dm = LocalDataManager(None)


rasterizer = build_rasterizer(cfg, dm)


# --- Function utils ---
# Original code from https://github.com/lyft/l5kit/blob/20ab033c01610d711c3d36e1963ecec86e8b85b6/l5kit/l5kit/evaluation/metrics.py
from torch import Tensor


def pytorch_neg_multi_log_likelihood_batch(
    gt: Tensor, pred: Tensor, confidences: Tensor, avails: Tensor
) -> Tensor:
    """
    Compute a negative log-likelihood for the multi-modal scenario.
    log-sum-exp trick is used here to avoid underflow and overflow, For more information about it see:
    https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
    https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    https://leimao.github.io/blog/LogSumExp/
    Args:
        gt (Tensor): array of shape (bs)x(time)x(2D coords)
        pred (Tensor): array of shape (bs)x(modes)x(time)x(2D coords)
        confidences (Tensor): array of shape (bs)x(modes) with a confidence for each mode in each sample
        avails (Tensor): array of shape (bs)x(time) with the availability for each gt timestep
    Returns:
        Tensor: negative log-likelihood for this example, a single float number
    """
    assert len(pred.shape) == 4, f"expected 3D (MxTxC) array for pred, got {pred.shape}"
    batch_size, num_modes, future_len, num_coords = pred.shape

    assert gt.shape == (batch_size, future_len, num_coords), f"expected 2D (Time x Coords) array for gt, got {gt.shape}"
    assert confidences.shape == (batch_size, num_modes), f"expected 1D (Modes) array for gt, got {confidences.shape}"
    assert torch.allclose(torch.sum(confidences, dim=1), confidences.new_ones((batch_size,))), "confidences should sum to 1"
    assert avails.shape == (batch_size, future_len), f"expected 1D (Time) array for gt, got {avails.shape}"
    # assert all data are valid
    assert torch.isfinite(pred).all(), "invalid value found in pred"
    assert torch.isfinite(gt).all(), "invalid value found in gt"
    assert torch.isfinite(confidences).all(), "invalid value found in confidences"
    assert torch.isfinite(avails).all(), "invalid value found in avails"

    # convert to (batch_size, num_modes, future_len, num_coords)
    gt = torch.unsqueeze(gt, 1)  # add modes
    avails = avails[:, None, :, None]  # add modes and cords

    # error (batch_size, num_modes, future_len)
    error = torch.sum(((gt - pred) * avails) ** 2, dim=-1)  # reduce coords and use availability

    with np.errstate(divide="ignore"):  # when confidence is 0 log goes to -inf, but we're fine with it
        # error (batch_size, num_modes)
        error = torch.log(confidences) - 0.5 * torch.sum(error, dim=-1)  # reduce time

    # use max aggregator on modes for numerical stability
    # error (batch_size, num_modes)
    max_value, _ = error.max(dim=1, keepdim=True)  # error are negative at this point, so max() gives the minimum one
    error = -torch.log(torch.sum(torch.exp(error - max_value), dim=-1, keepdim=True)) - max_value  # reduce modes
    # print("error", error)
    return torch.mean(error)


def pytorch_neg_multi_log_likelihood_single(
    gt: Tensor, pred: Tensor, avails: Tensor
) -> Tensor:
    """

    Args:
        gt (Tensor): array of shape (bs)x(time)x(2D coords)
        pred (Tensor): array of shape (bs)x(time)x(2D coords)
        avails (Tensor): array of shape (bs)x(time) with the availability for each gt timestep
    Returns:
        Tensor: negative log-likelihood for this example, a single float number
    """
    # pred (bs)x(time)x(2D coords) --> (bs)x(mode=1)x(time)x(2D coords)
    # create confidence (bs)x(mode=1)
    batch_size, future_len, num_coords = pred.shape
    confidences = pred.new_ones((batch_size, 1))
    return pytorch_neg_multi_log_likelihood_batch(gt, pred.unsqueeze(1), confidences, avails)


# ## Model

# Next we define the baseline model. Note that this model will return three possible trajectories together with confidence score for each trajectory.


class LyftMultiModel(nn.Module):

    def __init__(self, cfg: Dict, num_modes=3):
        super().__init__()

        architecture = cfg["model_params"]["model_architecture"]
        backbone = eval(architecture)(pretrained=True)
        self.backbone = backbone

        num_history_channels = (cfg["model_params"]["history_num_frames"] + 1) * 2
        num_in_channels = 3 + num_history_channels

        self.backbone.conv1 = nn.Conv2d(
            num_in_channels,
            self.backbone.conv1.out_channels,
            kernel_size=self.backbone.conv1.kernel_size,
            stride=self.backbone.conv1.stride,
            padding=self.backbone.conv1.padding,
            bias=False,
        )

        # This is 512 for resnet18 and resnet34;
        # And it is 2048 for the other resnets
        
        if architecture == "resnet50":
            backbone_out_features = 2048
        else:
            backbone_out_features = 512

        self.dropout = nn.Dropout(p=0.3)
            
        # X, Y coords for the future positions (output shape: batch_sizex50x2)
        self.future_len = cfg["model_params"]["future_num_frames"]
        num_targets = 2 * self.future_len
        other_agent_features = num_history_channels + 3 # extent info is 3d 
        total_num_features = backbone_out_features + other_agent_features
        self.head = nn.Linear(in_features=total_num_features, out_features=1024)

        # final prediction - a fc layer with desired number of outputs, no activation
        self.num_preds = num_targets * num_modes
        self.num_modes = num_modes
        self.logit = nn.Linear(1024, out_features=self.num_preds + num_modes)

    def forward(self, x, agent_data):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        
        # adding other agent data to image data
        x = torch.cat((x, agent_data), dim=1)

        # fc with relu activation and dropout
        x = self.dropout(F.relu(self.head(x)))
        x = self.logit(x)

        # pred (batch_size)x(modes)x(time)x(2D coords)
        # confidences (batch_size)x(modes)
        bs, _ = x.shape
        pred, confidences = torch.split(x, self.num_preds, dim=1)
        pred = pred.view(bs, self.num_modes, self.future_len, 2)
        assert confidences.shape == (bs, self.num_modes)
        confidences = torch.softmax(confidences, dim=1)
        return pred, confidences


def forward(data, model, device, criterion = pytorch_neg_multi_log_likelihood_batch):
    inputs = data["image"].to(device)
    target_availabilities = data["target_availabilities"].to(device)
    targets = data["target_positions"].to(device)
    bs = inputs.shape[0]
    history_positions = data['history_positions'].to(device).view(bs, -1)
    # centroid = data['centroid'].to(device).float()
    # yaw = data['yaw'].to(device).view(TRAIN_BATCH_SIZE, 1).float()
    # agent_data = torch.cat((history_positions, centroid, yaw, extent), dim=1)
    extent = data['extent'].to(device) / EXTENT_RANGE
    agent_data = torch.cat((history_positions, extent), dim=1)
    
    # Forward pass
    preds, confidences = model(inputs, agent_data)
    loss = criterion(targets, preds, confidences, target_availabilities)
    return loss, preds, confidences


# ==== INIT MODEL=================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'device {device}')


model = LyftMultiModel(cfg)


model.to(device)
optimizer = optim.Adam(model.parameters(), lr=cfg["model_params"]["lr"])
print(f'device {device}')


# load weight if there is a pretrained model
weight_path = cfg["model_params"]["weight_path"]
if weight_path != '':
    checkpoint = torch.load(weight_path)


print('weight file path is ', cfg['model_params']['weight_path'])


model.load_state_dict(checkpoint['state_dict'])

optimizer.load_state_dict(checkpoint['optimizer'])


# ## Evaluation

# # Evaluation
# 
# Evaluation follows a slightly different protocol than training. When working with time series, we must be absolutely sure to avoid leaking the future in the data.
# 
# If we followed the same protocol of training, one could just read ahead in the `.zarr` and forge a perfect solution at run-time, even for a private test set.
# 
# As such, **the private test set for the competition has been "chopped" using the `chop_dataset` function**.
# 
# ## DISCLAIMER
# **We're updating the dataset to support traffic lights. The code below has been designed to work with TLs, and it does not suppport the old interface. We expect the dataset to be online in the next few days** ( disclaimer added on 08/18/20)

# #### THIS CELL IS TO BE RUN ONLY ONCE TO GET VALIDATE_CHOPPED_100.ZARR DATASET FROM VALIDATE.ZARR
# ===== GENERATE AND LOAD CHOPPED DATASET

eval_base_path = cfg['data_path'] + 'scenes/validate_chopped_100'


# The result is that **each scene has been reduced to only 100 frames**, and **only valid agents in the 100th frame will be used to compute the metrics**. Because following frames in the scene have been chopped off, we can't just look ahead to get the future of those agents.
# 
# In this example, we simulate this pipeline by running `chop_dataset` on the validation set. The function stores:
# - a new chopped `.zarr` dataset, in which each scene has only the first 100 frames;
# - a numpy mask array where only valid agents in the 100th frame are True;
# - a ground-truth file with the future coordinates of those agents;

# In[47]:


eval_cfg = cfg["val_data_loader"]
eval_zarr_path = str(Path(eval_base_path) / Path(dm.require(eval_cfg["key"])).name)
eval_mask_path = str(Path(eval_base_path) / "mask.npz")
eval_gt_path = str(Path(eval_base_path) / "gt.csv")

eval_zarr = ChunkedDataset(eval_zarr_path).open()
eval_mask = np.load(eval_mask_path)["arr_0"]
# ===== INIT DATASET AND LOAD MASK
eval_dataset = AgentDataset(cfg, eval_zarr, rasterizer, agents_mask=eval_mask)
eval_dataloader = DataLoader(eval_dataset, shuffle=eval_cfg["shuffle"], batch_size=eval_cfg["batch_size"], 
                             num_workers=eval_cfg["num_workers"])
print(eval_dataset)


def model_validation_score(model, pred_path):
    # ==== EVAL LOOP
    model.eval()
    torch.set_grad_enabled(False)

    # store information for evaluation
    future_coords_offsets_pd = []
    timestamps = []
    confidences_list = []
    agent_ids = []
    progress_bar = tqdm(eval_dataloader)

    for data in progress_bar:

        _, preds, confidences = forward(data, model, device)

        #fix for the new environment
        preds = preds.cpu().numpy()
        world_from_agents = data["world_from_agent"].numpy()
        centroids = data["centroid"].numpy()
        coords_offset = []

        # convert into world coordinates and compute offsets
        for idx in range(len(preds)):
            for mode in range(3):
                preds[idx, mode, :, :] = transform_points(preds[idx, mode, :, :], world_from_agents[idx]) - centroids[idx][:2]

        future_coords_offsets_pd.append(preds.copy())
        confidences_list.append(confidences.cpu().numpy().copy())
        timestamps.append(data["timestamp"].numpy().copy())
        agent_ids.append(data["track_id"].numpy().copy())  
    
    write_pred_csv(pred_path,
               timestamps=np.concatenate(timestamps),
               track_ids=np.concatenate(agent_ids),
               coords=np.concatenate(future_coords_offsets_pd),
               confs=np.concatenate(confidences_list),
              )
    
    metrics = compute_metrics_csv(eval_gt_path, pred_path, [neg_multi_log_likelihood, time_displace])
    for metric_name, metric_mean in metrics.items():
        print(metric_name, metric_mean)
    
    #return [future_coords_offsets_pd, confidences_list, timestamps, agent_ids]


# Please note how `Num Frames==(Num Scenes)*num_frames_to_chop`. # 
# The remaining frames in the scene have been sucessfully chopped off from the data

# In[49]:


pred_path = f"{gettempdir()}/pred.csv"
model_validation_score(model, pred_path)
