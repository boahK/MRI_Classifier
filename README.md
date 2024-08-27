# 3D MRI Series Classifier
Official repository for "Classification of Multi-Parametric Body MRI Series Using Deep Learning", published in IEEE Journal of Biomedical and Health Informatics.

[[paper](https://ieeexplore.ieee.org/document/10645214)]


## Installation
You may need Conda environment. You can download Anaconda from this [link](https://www.anaconda.com/download).
Then, please run:
```
conda env create –f mriclassifier_env.yml
conda activate mriclassifier
```

## Dataset
This code runs using NIFTI (.nii.gz) files. Please put your NIFTI files into ./data.
For preprocessing data, please run:
```
python data_preprocessing.py
```
Then, the processed data will be saved in ./data/preprocessed.

## Train

To train our model, run the following command:

```train
sh train.sh
```
The checkpoints of the model will be saved in ./checkpoint/.

## Test

To test the trained our model, run:

```eval
sh test.sh
```
The results of the MRI classification such as the confusion matrix will be saved in ./results.

To use our pre-trained model, please download the model weights from [here]() and put the weights in ./pretrained_model.

