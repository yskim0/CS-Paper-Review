# DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs

## Overview

DCNNs (Deep Convolutional Neural Networks)  image classification, object detection
- built-in invariance to local image transformations: learn abstract data representation
 // semantic segmentation: needs dense prediction tasks

> [ Problems and Solutions ]
1. reduced feature resolution: Atrous Convolution (Dilated Convolution)
- repeated combination of max-pooling and downsampling (‘striding’)
- remove the downsampling operator from the last few max pooling layers of DCNNs and instead upsample the filters in subsequent convolutional layers by inserting holes between non-zero filter taps with zero values.
- combine combination of atrous convolution followed by simple bilinear interpolation of the feature responses to the original image size.
 enlarge the field of view of filters without increasing the number of parameters or the amount of computation.

2. existence of objects at multiple scales: Atrous Spatial Pyramid Pooling (ASSP) with different rates
- resampling a given feature layer at multiple rates prior to convolution. = multiple parallel atrous convolutional layers with different sampling rates
- multiple filters that have complementary effective fields of view  capturing objects as well as useful image context at multiple scales

3. reduced localization accuracy: combining DCNNs with CRFs (Conditional Random Fields)
- CRFs : combine class scores computed by multi-way classifiers with the low-level information captured by the local interactions of pixels and edges, or superpixels.
 efficient computation and ability to capture fine edge details while also catering for long range dependencies (=connections).

> [ DeepLab model ]
- transform all the fully connected layers to convolutional layers
– increase feature resolution through atrous convolutional layers – employ bi-linear interpolation to upsample the score map to reach the original image resolution – yield the input to a fully-connected CRF that refines the segmentation results

## Related Work

전체적으로 DeepLab은 semantic segmentaion을 잘 해결하기 위한 방법으로 atrous convolution을 적극적으로 활용할 것을 제안하고 있다.
V1에서는 atrous convolution을 적용해 보았고, 
V2에서는 multi-scale context를 적용하기 위한 Atrous Spatial Pyramid Pooling (ASPP) 기법을 제안하고, 
V3에서는 기존 ResNet 구조에 atrous convolution을 활용해 좀 더 dense한 feature map을 얻는 방법을 제안하였다.
그리고 최근 발표된 V3+에서는 separable convolution과 atrous convolution을 결합한 atrous separable convolution의 활용을 제안하고 있다.

## Methods

1. Atrous Convolution
2. Atrous Spatial Pyramid Pooling (ASSP)
3. Fully Connected CRFs (Conditional Random Fields)

## Additional Studies

DeepLab v1

## Reference

1. fully convolutional network for semantic segmentation
2. Multi-Scale Context Aggregation by Dilated Convolutions (DeepLab v1)
3. Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRFs
