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

## Lyft l5kit library
[# l5kit library](https://github.com/lyft/l5kit)
1. Library containing APIs to handle Lyft datasets
2. Includes functions to read .zarr files, parse them to useful information 
   and many visualisation features
3. Currently two examples - visualisation and sample submission notebooks are 
   part of github repo to help get started
