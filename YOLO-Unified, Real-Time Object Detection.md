# Reports for Deep Learning Society of undergraduates in Ewha

### Date: July 31, 2020
### Student ID number / Name: 1785044 / 김연수
### Name of the thesis: 

[You Only Look Once : Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640)


## Overview
(You should include contents of summary and introduction.)

**기존**의 object detection = repurpose classifiers to perform detection


- DPM : sliding window
- R-CNN : region proposal
- Classifier를 통해 클래스 분류 -> post-processing 통해 refine -> rescore 후 합침


**YOLO** : a single network, end-to-end directly on detection performance


- Unified Model (feat. GoogLeNet, NIN)
- Real-time 가능할 정도로 빠름.
- Single regression problem 
    - Reasons globally about the image when making predictions
    - bbox coord. & class probability 동시에 구할 수 있음.
    - CNN -> Non-max. supprestion -> Finish!
- Can learns generalizable representations of obj. 
- 다양한 datasets 가능하다.


## Related work (Basic concepts)

GoogLeNet architecture


NIN(Network in Network)
-	네트워크를 구성할 때 또 다른 micro network 포함하여 설계했을 때 성능 향상이 됨.


## Methods
(Explain one of the methods that the thesis used.)

- Unified Detection 
    - 모든 bbox, class를 한번에 고려한다 (=> 즉, 전체 이미지로 한번 본다)
    
    <img width="460" alt="스크린샷 2020-07-31 오전 2 46 20" src="https://user-images.githubusercontent.com/48315997/88956229-05c84000-d2d8-11ea-9878-e3d7baf029a4.png">


    ```
    1. S x S grid
    2. bbox & confidence score(bbox에서 물체가 많이 포함되는지)
        - confidence = Pr(object) * IOU
        - no obj -> 0(zero)
        - each bbox contains : x,y,w,h,confidence
        - x, y : bbox안의 cell 위치 (norm. 0~1)
        - w, h : bbox width, height (norm. 0~1)
        - c : bbox confidence
    3. Class probability
    ```

    - Pr(class(i)|Object) * Pr(Object)
    - Predictions are encoded as an S x S x (B*5 + C)
    - On Pascal Voc, S=7, B=2. C(class) = 20
        - Thus, 7 x 7 x 30 tensor.



    - **Architecture**
<br><img width="742" alt="스크린샷 2020-07-31 오전 2 46 53" src="https://user-images.githubusercontent.com/48315997/88956268-18427980-d2d8-11ea-86fa-65b0290687e6.png">

        - GoogLeNet을 응용
        - 24 conv. layer + 2 FC layer
    	    - Fast YOLO(tiny YOLO) : 9 conv layer 사용함.
        - GoogLeNet에서 쓰이는 Inception module 대신, Reduction layer(1x1 conv)를 사용하여 파라미터 size 줄임.
        - Feature Extractor에 해당하는 20 conv layer로 pretrain함.
            - Pretrained layer 바탕으로 VOC data에 fine-tuning
        - 좀 더 좋은 detection을 위해 224 -> 448로 size 2배 늘림.
        - NIN(network in network) - 4 conv layer, 2 FC layer
=> classifier 역할 



    - Loss function
    <br><img width="424" alt="스크린샷 2020-07-31 오전 2 49 08" src="https://user-images.githubusercontent.com/48315997/88956501-69526d80-d2d8-11ea-8934-475ab7c5c5bf.png"> 


        - MSE보다 SSE가 단순하니 SSE로 선택
        - Object의 유무는 grid cell 자체에서 cell 기준으로 classify하므로 유무를 판단하는 loss보다는 bbox 좌표를 찍는 loss를 더 크게 봄.
            - lambda(coord) = 5 and lambda(noobj) = 0.5
            - bbox 중에서도 larger box는 오차로 인한 로스 변동이 클 것이므로 루트를 씌워서 작은 bbox에는 전보다 더 크게 반응, 큰 bbox에는 전보다 조금 더 작게 반응하도록 함.
        - 1(obj,i) : if object appears in cell i
    	- 1(obj,ij) : the j-th bbox predictor in cell i is “responsible” for that prediction


    - Limitations
    	- Bbox가 grid size에 의존.
            - Image가 domain에 자주 나타나는 object size를 제대로 반영하지 못한다면 bad
        - Grid size보다 작은 물체가 있을 경우 bad


- experiment rusults

<img width="440" alt="스크린샷 2020-07-31 오전 2 53 35" src="https://user-images.githubusercontent.com/48315997/88956905-090ffb80-d2d9-11ea-958d-be5f22900922.png">

<img width="446" alt="스크린샷 2020-07-31 오전 2 54 10" src="https://user-images.githubusercontent.com/48315997/88956952-1cbb6200-d2d9-11ea-8f84-b1b67a802ae8.png">

    - bg 관련해서 덜 실수함.

- Fast R-CNN과 combine했더니 속도 떨어짐

<img width="453" alt="스크린샷 2020-07-31 오전 2 54 32" src="https://user-images.githubusercontent.com/48315997/88956988-293fba80-d2d9-11ea-96db-5277a1212d16.png">

 
- 새로운 dataset에도 잘 작동함.

 	 
 

## Code

- pytorch 

- network implementation

    ```python
    # feature extractor 부분 
        feature_extract_net = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 192, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(192, 128, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 256, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(512, 256, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 256, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 256, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 256, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 512, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(1024, 512, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(1024, 512, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )
    ```

    ```python
    # classifier 부분
    conv = nn.Sequential(
                # 4 conv layer
                nn.Conv2d(1024, 1024, 3, padding=1),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(1024, 1024, 3, stride=2, padding=1),
                nn.LeakyReLU(0.1),

                nn.Conv2d(1024, 1024, 3, padding=1),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(1024, 1024, 3, padding=1),
                nn.LeakyReLU(0.1, inplace=True)
                Flatten(),

                # 2 FC layer
                nn.Linear(7 * 7 * 1024, 4096),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Dropout(0.5, inplace=False), 
                nn.Linear(4096, S * S * (5 * B + C)),
                nn.Sigmoid()
        )
    ```

## Additional studies
(If you have some parts that cannot understand, you have to do additional studies for them. It’s optional.)

NIN(network in network)


## References
(References for your additional studies)

https://www.youtube.com/watch?v=eTDcoeqj1_w
https://arclab.tistory.com/162
https://arclab.tistory.com/165


