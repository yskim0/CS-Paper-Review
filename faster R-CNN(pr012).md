# Reports for Deep Learning Society of undergraduates in Ewha


## Date: July 22, 2020


## Student ID number / Name: 1785044 / 김연수


## Name of the thesis: Faster R-CNN : Towards Real-Time Object Detection with Region Proposal Networks


## Overview
(You should include contents of summary and introduction.)


* 흐름 : R-CNN -> (SPP-net) -> Fast R-CNN -> Faster R-CNN


1. R-CNN : Region with CNN features. RoI 들을 뽑아내고 CNN에 각각 집어넣음.
    - Region Proposal – Selective Search : 어떻게 바운딩 박스(bbox)를 뽑아내는가
    - Training 
    - Pre-train AlexNet
    - SVM, bounding-box regressor (CNN에 학습이 안된다는 단점이 있음)


2. Fast R-CNN
    - RoI projection -> 매 bounding box마다 RoI pooling
    - 어떤 RoI가 나와도 똑같은 size가 나오도록 max pooling => RoI pooling
    - Fixed-length feature vector from RoI가 됨.
    - FC 레이어에 넘기면서 classification(K+1 class) + bounding box location 동시에 계산
    - Problems of Fast R-CNN : Out-of-network region proposals are the test-time computational bottleneck


3. Faster R-CNN
    - Notion : Region Proposal을 Selective Search(CPU에서 했음)을 하지 말고 실제 네트워크 안에서 같이 해보자(GPU로 계산 가능)
    - **Key Point : Region Proposal Network(RPN) + Fast R-CNN**
    (사진)
    - CNN을 share하는 것을 목표로, 즉 네트워크가 하나인 것처럼 해보자!
    (사진)
    - RPN (사진)
    - Loss function (사진)
    - 4-step Alternation Training (여기 사진)



## Summary of experiments
(Explain figures briefly.)

(사진)
Proposal time 매우 줄어듦!


## Methods
(Explain one of the methods that the thesis used.)


Region Proposal을 실제 네트워크 안에서 같이 해보자
Region Proposal Network + Fast R-CNN 
CNN share하자 

등 Overview 에서 설명함


## Ideas for further research
(Explain your ideas for further research briefly.)


Faster R-CNN의 한계 : RoI pooling 할 때 7의 배수가 아닌 RoI가 나오면 버림이 있을 것
- 오차 발생… (object detection에는 큰 문제가 아닐 수 있지만 location 등이 중요한 것에는 문제를 일으킬 수 있음)
- Mask R-CNN 으로 극복

아직도 실시간으로 처리하기에는 조금 부족해 보이기도…
- Proposal 시간은 유의미하게 줄이기 힘들 것 같음. Region-wise time을 좀 더 줄이는 방법? Conv도?...


## Additional studies
(If you have some parts that cannot understand, you have to do additional studies for them. It’s optional.)

R-CNN
Fast R-CNN
Mask R-CNN

## References
(References for your additional studies)

https://curt-park.github.io/2017-03-17/faster-rcnn/

유튜브(링크)

## Code
```
