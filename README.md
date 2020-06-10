# The implement of MDL model for segmentation based on few short learning

## Introduction

Few-shot semantic segmentation methods aim for predicting the regions of different object categories with only a few labeled samples. It is difficult to produce segmentation results with high accuracy when a new category appears. In this paper, we propose a Multi-scale Discriminative Location-aware (MDL) network to tackle the few-shot semantic segmentation problem. In order to use information from different levels, we first keep the last three convolutional layers of FCN, and then use the VGG-16 network to extract features from the
support image-label pair, which adjusts the weight of the query image segmentation branch. Discriminative location-aware architecture can improve the efficiency of few-shot segmentation, and therefore the global average pooling layer is added to produce location feature information. Finally, we evaluate our MDL model on the Pascal VOC 2012 challenge, and show that it achieves competitive mIoU score compared to methods in recent years.
