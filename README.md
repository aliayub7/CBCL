# Cognitively-Inspired Model for Incremental Learning Using a Few Examples
Keras code for the paper Cognitively-Inspired Model for Incremental Learning Using a Few Examples https://arxiv.org/abs/2002.12411
## Applied on CIFAR-100, Caltech-101 and CUBS-200-2011 

### Requirements
* Keras (Version 2.1.6)
* Scipy (Currently working with 1.2.1)
* Scikit Learn (Currently working with 0.21.2)
* if you can't find a pre-trained model in Kears, get them from from https://github.com/qubvel/classification_models
* Download the datasets in */data directory
## Usage
* First run ```get_features.py``` to get the ResNet features for all the images in the dataset. For Caltheh-101 and CUBS-200-2011 use ResNet-18 instead of ResNet-34.
* After feature extraction, simply run ```main_file.py``` to get the results for all increments.
## If you consider citing us
```
@InProceedings{Ayub_2020_CVPR_Workshops,  
author = {Ayub, Ali and Wagner, Alan R.},  
title = {Cognitively-Inspired Model for Incremental Learning Using a Few Examples},  
booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},  
month = {June},  
year = {2020}  
}
```

