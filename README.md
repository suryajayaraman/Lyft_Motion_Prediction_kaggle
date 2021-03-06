# Lyft_Motion_Prediction_kaggle
Repo for the Kaggle competition - Level 5 Lyft Motion prediction

## Kaggle competition rules an know how
1. This Competition rules allow submissions only in kaggle notebook format
2. Has restrictions on Accelerator run times (GPS ~9hrs, TPUs ~9 hrs)
3. Accelerators work with model training and inference only, not during
   other operations such as np arrays, df etc
4. Use of Accelerators reduces no of CPU cores 

## Dataset
Kaggle has roughly 30% of the entire dataset (~22 GB) which is currently used
for training the model.Entire dataset can be found at Lyft L5 official site, 
which is around 71 GB

### Folders in Dataset
1. Aerial map- aerial_map.png
2. scenes - sanple.zarr, train.zarr, test.zarr, validate.zarr
3. semantic map - semantic_map.pb

## Lyft l5kit library
[# l5kit library](https://github.com/lyft/l5kit)
1. Library containing APIs to handle Lyft datasets
2. Includes functions to read .zarr files, parse them to useful information 
   and many visualisation features
3. Currently two examples - visualisation and sample submission notebooks are 
   part of github repo to help get started


## lyft-kaggle-visualisation.ipynb

Main packages in l5kit 
1. Rasterisation (convert .zarr files to multi channel tensors, images for visualisation)
   Each class contains atleast two functions - rasterise, to_rgb

2. Visualisation - contains utilites for drawing on RGB images

3. Dataset - EgoDataset and AgentDataset classes to covnert rasterised images to multi-
   channel images

Here Zarr dataset contains 
1. frames - timestamp, some id, ego translation and rotation info
2. agents - centroid, extent, yaw, velocity, trackid, label_probabilities
3. scenes 
4. tl-faces 

## lyft_motion_prediction_kernels

1. Public notebooks in competition and their summary

## Lyft_motion_prediction_discussion

1. Important topics discussed in competition
2. l5kit utility fix, l5kit library convention and understanding
3. Links to research papers and articles
4. Notable approaches to solving the problem - running pytorch with TPUs, GCS integration, baseline models etc

## Resnet baseline

1. At each instance, rasterized image data (containing ego and agent past trajectories, traffic light info, lane info etc) is passed through
Resnet models (Resnet18, Resnet34, Resnet50)
2. Final softmax layer is replaced by linear fully connected layer
3. Negative log likelihood function is used as loss function 
4. Adam optimizer with fixed learning rate used as of now
3. Varying parameters include
	- Number of history frames
	- Raster image size
	- pixel resolution 

## LSTM baseline

Since this is a sequence problem, I tried to implement LSTM based model. The approach is as follows: 

[LSTM_baseline_idea](submissions/LSTM_Baseline/lstm baseline idea.jpg)

Implementation can be found at this [kaggle kernel](https://www.kaggle.com/suryajrrafl/lstm-baseline)


## Argoverse references

Similar competition was held based on Argoverse Motion prediction dataset and the reference repo for that competition
can be found at this link [Argoverse Git repo](https://github.com/jagjeet-singh/argoverse-forecasting)_
