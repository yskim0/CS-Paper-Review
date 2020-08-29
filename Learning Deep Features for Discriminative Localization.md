# Learning Deep Features for Discriminative Localization

## Overview

**1. CNN이 어떤 데이터의 정보를 중요하게 생각해 최종적인 결과를 냈는지 알아보고,**   
` Visualize class discriminative features(CNN) `   
**2. 이를 이용해 CNN을 classification 뿐만 아니라, detector로 작용해 discriminative image region을 찾는, localization 기능을 수행한다.**   
` Weakly-supervised Learning for Object Localization `   
<br>

> CNN: 위치 정보에 대한 supervision 없이도 object detectors로 작용한다.   
> 하지만 Fully-Connected Layers가 적용되면서 classification에 적합해지고 **위치정보가 flatten되면서 detector의 기능을 상실한다.**   
<br>

### CAM (Class Activation Map)
![archi](https://camo.githubusercontent.com/fb9a2d0813e5d530f49fa074c378cf83959346f7/687474703a2f2f636e6e6c6f63616c697a6174696f6e2e637361696c2e6d69742e6564752f6672616d65776f726b2e6a7067)   
1. global average pooling을 이용해 structural regularizing: overfitting 방지
  - 마지막 convolutional layer를 분류해야 하는 class의 개수만큼 채널 수를 설정한다.   
    _(최종 feature map의 채널 수 == class 개수)_   
  - 최종 feature map에서 채널 별로 average를 구한 뒤, 가장 큰 값을 가지는 부분을 pooling한다.   
    _(각각의 max 값이 class에 대응하는 값으로 pooling 됨)_   
2. fully-connected layers 최소화
  - GAP를 이용해 최종 FC layer을 제외한 FC layer들을 모두 제거한다. 따라서 파라미터의 수를 줄여 computation의 효율을 높이고, detector로서의 기능을 복구한다.

```
attention-based model instantly by tweaking your own CNN   
: highlights the most informative image regions relevant to the predicted class
```

## Related Work

## Methods

### GAP (Global Average Pooling)
![overall](https://you359.github.io/images/contents/cam_CNNwithGAP.png)   
1. feature map(중 하나)인 f_k(x,y)는 GAP를 통해 F_k로 summation 된다.   
2. 이를 CNN의 마지막 layer인 fully connected layer로 전달하면서 w_k_c * f_k(x,y)- linear combination(== weighted sum)- 을 구한다.
3. S_c를 softmax layer에 전달하여 최종 output을 pooling한다.   
<br>

이 때, weighted sum을 M_c(x,y)로 정의할 수 있는데, 이는 클래스 c에 대한 Map이다.   
![map](https://you359.github.io/images/contents/cam_what-is-cam.png)   
_CAM이 특정 클래스 c를 구별하기 위해 CNN이 어떤 영역을 주목하고 있는지 시각화하는 방법이므로,   
해당 Map의 집합(== 이를 average pooling한 값인 S_c)이 CAM을 의미한다._   
<br>

#### 1. GAP vs GMP   
![gap_gmp](https://you359.github.io/images/contents/cam_gap.png)   
| GMP | GAP |
|------|-----|
|identify just one discriminative part|identify the extent of the object|
|: low scores for all image regions except the most discriminative one do not impact the score but performing a max|averaging makes the value to be maximized by finding all discriminative parts of an object while all low activations reduce the output of the particular map|
<br>

#### 2. FC vs GAP
![fc_gap](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FboM0El%2FbtqBtGTWdxd%2FT3SfcjlZ9mk1uFsirLkLT0%2Fimg.png)   
| FC | GAP |
|------|-----|
|parameter 개수 많음|parameter 개수 적음 => regularizer 역할, 과적합 방지|
|flatten == 위치 정보 손실|1 * 1 node == 위치 정보 유지|
<br>

## Code

## Reference
