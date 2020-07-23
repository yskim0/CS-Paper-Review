# Reports for Deep Learning Society of undergraduates in Ewha


## Date: 
July 23, 2020


## Student ID number / Name: 
1785044 / 김연수


## Name of the thesis: 
Faster R-CNN : Towards Real-Time Object Detection with Region Proposal Networks

(https://arxiv.org/pdf/1506.01497.pdf)

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
    


    - CNN을 share하는 것을 목표로, 즉 네트워크가 하나인 것처럼 해보자!
      
      <img width="400" alt="그림1" src="https://user-images.githubusercontent.com/48315997/88275099-c61db900-cd17-11ea-9e61-b0f44172936a.png">


    - RPN 
    ![IMG_5BDD1471DB01-1](https://user-images.githubusercontent.com/48315997/88275179-e6e60e80-cd17-11ea-9f04-8a258dc4778d.jpeg)    


    - Loss function (사진)
    
        ![IMG_7BFAB2EF8A93-1](https://user-images.githubusercontent.com/48315997/88275199-f2d1d080-cd17-11ea-8a88-53902efa4790.jpeg)
        
    - 4-step Alternation Training 

        



## Summary of experiments
(Explain figures briefly.)

<img width="500" alt="스크린샷 2020-07-23 오후 7 10 42" src="https://user-images.githubusercontent.com/48315997/88275353-39bfc600-cd18-11ea-967f-4799ddcbba59.png">



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

https://www.youtube.com/watch?v=kcPAGIgBGRs&list=PLXiK3f5MOQ760xYLb2eWbtOKOwUC-bByj&index=13&t=0s

https://curt-park.github.io/2017-03-17/faster-rcnn/


## Code

구현이 복잡하기 때문에 추후로 미룸
```
