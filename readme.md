[comment]: <> (# RSNA Screening Mammography Breast Cancer Detection Competition Solution)

<!-- PROJECT LOGO -->

<p align="center">

  <h1 align="center"> RSNA Screening Mammography Breast Cancer Detection Competition Solution
  </h1>
  <p align="center">
   <strong>Oscar Chan</strong>
  </p>

<p align="center">
This is my solution for <a href=https://www.kaggle.com/c/rsna-miccai-brain-tumor-radiogenomic-classification/overview>RSNA Screening Mammography Breast Cancer Detection Competition</a>. The goal of this competition is to develop a machine learning model to classify breast cancer using mammography images. 
</p>
<br>

# Getting Started
## Installation
Install the dependencies by running the following command:
```bash
pip install -r requirements.txt
```

## Run

To run and track the training progress, you may use the following command:
```bash
wandb login --verify ## Your Wandb api key ##
python main.py
```
Or you may run the code with jupyter notebook/colab/kaggle with the file named main.py

## Data
The data is available on [Kaggle](https://www.kaggle.com/c/rsna-miccai-brain-tumor-radiogenomic-classification/data). You can download the data and put it in the `data` folder.

Set `cfg.data_dir` in `config.py` to the path of the data folder. The dataset class is available in `Dataset.py`

## Model
The model calls API from timm, which is a pytorch model library. The model class is available in `Model.py`

## Dicom to Image Preprocessing
It is recommended to use available kaggle dataset for this part, for more customization, you may refer to `Dicom2Image.ipynb`. 

Since Dicom images are encoding in more than 8 bits, thus it is necessary to apply VOI LUT function to select important colour ranges of the images, and then convert the images to 8 bits which is more suitable for observation and possibily easierfor training. 

In this step, image is resized to running size(512x256/1024x512), and then saved as png file, this will reduce computational cost and running time for training.

## Training
This project implements a trainer for training the model. The trainer is available in `Trainer.py`. You can set the hyperparameters in `config.py` and run 
```bash
python main.py
```
to train and evaluate the model.

There are serveral input parameters in trainer you need to specify:
1. `cfg`: the config file
2. `df`: the dataframe of the entire dataset(i.e. cfg.data_df), it is first preprocessed in `main.py` before passing into the trainer
3. `model`: the model you want to train
4. `scaler`: the scaler you want to use to scale the gradient
5. `loss_calculation`: when there are multiple losses, you can specify how to calculate the loss
6. `loss_functions`: the loss functions for each output class you want to use
7. `test`: whether it is in test mode, When `test` is True, val_df will become `test_df` 
8. `test_df`: the dataframe of the test dataset, it is first preprocessed in `main.py` before passing into the trainer

To fit the model after initializing the trainer as `ter`, you may run:
```python
best_metrics = ter.fit()
```
The best model will be saved according to best thresholded F1 score and loss to `cfg.model_dir`. The best model(according to best thresholded F1 score) will be returned.

To evaluate the model, you may run:
```python
train_df = ter.predict("Train", best=True)
val_df = ter.predict("Val", best=True)
train_df.to_csv(f"train.csv", index=False)
val_df.to_csv(f"val.csv", index=False)

train_wandb_table = wandb.Table(dataframe=train_df)
val_wandb_table = wandb.Table(dataframe=val_df)
wandb.run.log({"Eval on Train": train_wandb_table})
wandb.run.log({"Eval on Val": val_wandb_table})

get_probability_hist([train_df, val_df], bins=30)
get_corr_matrix([train_df, val_df])
get_PR_curve([train_df, val_df], mode="single", best_metric=ter.best_metric)
```
## Inference
To inference the model, you may run `Inference.ipynb` on kaggle to infer on the hidden test dataset.

## Results
Best Models score summary:\
• Public LB: \
• Private lb: \
• Local Score: \

## 



