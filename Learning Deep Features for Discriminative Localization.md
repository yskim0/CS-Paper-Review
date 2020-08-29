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

> **CAM에서 activate된 부분은 모델이 object를 판단한 기준일 뿐, object가 아니다.**   
<br>

## Related Work
1. Weakly Supervised Object Localization using CNN
  + self-taught object localization involving masking out image regions to identify the regions causing the maximal activations in order to localize objects   
  + combine multiple-instance learning with CNN features to localize objects   
  + transferring mid-level image representations and show that some object localization can be achieved by evaluating the output of CNNs on multiple overlapping patches   
  + global max-pooling to localize a point on objects
2. Visualizing CNNs
  + use deconvolutional networks to visualize what patterns activate each unit   
  + same network can perform both scene recognitiona and object localization in a single forward-pass   
  + analyze the visual encoding of CNNs by inverting deep features at different layers   

## Methods

### GAP (Global Average Pooling)
![overall](https://you359.github.io/images/contents/cam_CNNwithGAP.png)   
1. feature map(중 하나)인 f_k(x,y)는 GAP를 통해 F_k로 summation 된다.   
2. 이를 CNN의 마지막 layer인 fully connected layer로 전달하면서 w_k_c * f_k(x,y)- linear combination(== weighted sum)- 을 구한다.
3. S_c를 softmax layer에 전달하여 최종 output을 pooling한다.   
<br>

이 때, weighted sum을 M_c(x,y)로 정의할 수 있는데, 이는 클래스 c에 대한 Map이다.   
![map](https://you359.github.io/images/contents/cam_what-is-cam.png)   

```
CAM이 특정 클래스 c를 구별하기 위해 CNN이 어떤 영역을 주목하고 있는지 시각화하는 방법이므로,   
해당 Map의 집합(== 이를 average pooling한 값인 S_c)이 CAM을 의미한다.   
```
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

## Results

#### CAM
![CAM](https://camo.githubusercontent.com/c9806e2dfb8e60780258305ccf1c5fe3973cccc0/687474703a2f2f636e6e6c6f63616c697a6174696f6e2e637361696c2e6d69742e6564752f6578616d706c652e6a7067)   

#### Classification Error
![classify](https://miro.medium.com/max/2920/1*9oq21Z--PU6nh18HZ6KuKg.png)   

#### Weakly-supervised Object Localization   
1. CNN모델(AlexNet, VGGNet, GoogLeNet)에 중간 FC layer를 빼고 GAP을 넣는 등 변형   
2. 해당 모델을 classification 바탕으로 학습시킨 후 CAM을 추출해 상위 20% segment 후 가장 큰 object를 detect한 bounding box 생성
  `classification dataset으로 object localization 수행`   
![weak_result](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Froy6R%2FbtqBADINhB5%2F1MbxqX2IgEJFv743UTPouK%2Fimg.png)   
> * backprop보다 좋은 성능   
> * full-supervised network인 AlexNet과 비슷한 성능

#### Deep Features for Generic Localization
![feat](https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcTDqBHSLfThnxf2NbSnP5If7IYnY_pth47WzA&usqp=CAU)   
_generic discriminative localization using GoogLeNet-GAP deep features_   

![inform](https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcT8oGGpSMpQ2QNoEGkIXkKOOjUuRJvOyMd2bg&usqp=CAU)   
_discovering informative objects in the scenes_   

![weak](https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcTAtQbl8J6346HOzzUDWPGkux7FwcDi_5zrrw&usqp=CAU)   
_concept localization in weakly labeled images_   

![text](https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQ_FFXj0b7CewjW4qnT6jTOpEvpRuV1bVip8w&usqp=CAU)   
_weakly supervised text detector_   

![answer](https://i.imgur.com/NfsVB9o.png)   
_interpreting visual question answering_

#### Visualizing Class-Specific Units
![unit](https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcRxBqNUNmAl-GZL_YyYF3Ps4SX-CkbIJ-Kd6g&usqp=CAU)   

## Code

### CAM
_https://github.com/zhoubolei/CAM_   
<br>

1. 마지막 Convolution layer의 출력인 ‘last_conv_output’   

```
    model_input = model.input
    model_output = model.layers[-1].output

    # f_k(x, y) : 마지막 Convolution layer의 출력 feature map
    f_k = model.get_layer(last_conv).output

    # model의 입력에 대한 마지막 conv layer의 출력(f_k) 계산
    get_output = K.function([model_input], [f_k])
    [last_conv_output] = get_output([img_tensor])

    # batch size가 포함되어 shape가 (1, width, height, k)이므로
    # (width, height, k)로 shape 변경
    # 여기서 width, height는 마지막 conv layer인 f_k feature map의 width와 height를 의미함
    last_conv_output = last_conv_output[0]
여기서 K.function은 keras.backend.function으로, placeho
```

2. linear combination(weighted sum)을 위한 해당 클래스에 대한 weight들   

```
    # 출력(+ softmax) layer와 GAP layer 사이의 weight matrix에서
    # class_index에 해당하는 class_weight_k(w^c_k) 계산
    # ex) w^2_1, w^2_2, w^2_3, ..., w^2_k
    class_weight_k = model.layers[-1].get_weights()[0][:, class_index]
    
    # 마지막 conv layer의 출력 feature map(last_conv_output)과
    # class_index에 해당하는 class_weight_k(w^c_k)를 k에 대응해서 linear combination을 구함

    # feature map(last_conv_output)의 (width, height)로 초기화
    cam = np.zeros(dtype=np.float32, shape=last_conv_output.shape[0:2])
    for k, w in enumerate(class_weight_k):
        cam += w * last_conv_output[:, :, k]
```

## Reference
https://you359.github.io/cnn%20visualization/CAM/   
https://kangbk0120.github.io/articles/2018-02/cam   
https://dambi-ml.tistory.com/5   
