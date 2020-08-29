# Reports for Deep Learning Society of undergraduates in Ewha

### Date: Aug. 29, 2020
### Student ID number / Name: 1785044 / 김연수
### Name of the thesis: 

[NEURAL ARCHITECTURE SEARCH WITH REINFORCEMENT LEARNING](https://arxiv.org/abs/1611.01578)


## Overview
(You should include contents of summary and introduction.)

> we use a re- current network to generate the model descriptions of neural networks and train this RNN with reinforcement learning to maximize the expected accuracy of the generated architectures on a validation set.

RNN을 이용해서 neural network의 model description(하이퍼 파라미터: # of filters, stride length ...)을 문자열로 생성한다.
강화학습을 통해 expected accuracy를 최대로 만든다.

<img src="https://miro.medium.com/max/658/1*KgICs1DPpGbqY2WWPn1kwg.png" width = 500>

1. Controller에서 p의 확률로 A라는 Architecture를 생성한다.
2. 자식 네트워크에서는 A 아키텍쳐를 훈련시켜 정확도 R을 구한다.
3. 정확도를 리워드의 신호로 사용한다. `policy gradient`를 계산해서 컨트롤러를 업데이트한다.
4. 반복하다보면 더 높은 확률로 더 높은 정확도를 보이는 아키텍쳐를 찾을 수 있다.



## Related work (Basic concepts)

NAS 부분의 거의 최초라고 볼 수 있음.

이전 연구들 : Hyperparameter optimization
> it is difficult to ask them to **generate a variable-length configuration** that specifies the structure and connectivity of a network

유전자 알고리즘
- search-based 방식이라 탐색속도가 느림.

컨트롤러에서의 Neural Architecture 방식은 이전 예측값들을 input으로 받아 하이퍼 파라미터를 한 번에 하나씩 예측하는 `auto-regressive`한 방식이다.


## Methods
(Explain one of the methods that the thesis used.)

이 논문의 Key point : **skip connection 예측하여 모델의 복잡도를 높인 것**, **파라미터 접근방식을 사용해서 훈련 속도를 높인 것**

1. Generate Model with a Controller Recurrent Neural Network

<img src="https://miro.medium.com/max/700/1*9dgifjZ6BKyPqzIxR-EGwg.png" >

> It predicts filter height, filter width, stride height, stride width, and number of filters for one layer and repeats. Every prediction is carried out by a softmax classifier and then fed into the next time step as input.

컨트롤러를 이용하여 CNN 모델에 사용하는 하이퍼파라미터들을 생성함.

레이어마다 사용할 필터, Stride 값을 예측하고 반복함.

**하이퍼 파라미터 예측시에 softmax classifier를 거친값이 다음 스텝의 input으로 들어감.**


컨트롤러 RNN이 아키텍쳐를 생성하면 생성된 아키텍쳐의 뉴럴 네트워크를 훈련시킴.


> The parameters of the controller RNN, θc, are then optimized in order to maximize the expected validation accuracy of the proposed architectures.

Validation set으로 네트워크의 정확도를 측정하고, 컨트롤러 RNN의 파라미터 세타C는 정확도의 기대값을 최대화하기 위해 최적화됨.

2. Training with Reinforce

![2](https://miro.medium.com/max/456/1*T03ptXjYcHkOBLFfoEs89A.png)
- controller to maximize its expected reward

- 컨트롤러 token list a[1]:a[T] : Architecture predicted by the controller RNN viewed as a sequence of actions

- 자식 네트워크는 생성된 구조의 정확도 R을 출력하고, 이 R을 강화학습의 리워드로 사용해서 컨트롤러를 강화학습 훈련시킴.
- Layer 하나짜리 CNN에서의 T=3임.
    - a1 : filter height
    - a2 : filter width
    - a3 : # of filters

![스크린샷 2020-08-29 오전 11 55 12](https://user-images.githubusercontent.com/48315997/91626958-815c0080-e9ee-11ea-89cb-43ee634d253e.png)

![스크린샷 2020-08-29 오후 12 11 09](https://user-images.githubusercontent.com/48315997/91627195-bc5f3380-e9f0-11ea-9bd2-3d2b8653075c.png)

![스크린샷 2020-08-29 오후 12 11 30](https://user-images.githubusercontent.com/48315997/91627201-c84af580-e9f0-11ea-8824-88e3e1f87255.png)

> In this work, we use the REINFORCE rule from Williams (1992)

- Standard REINFORCE Update Rule
- R은 미분 불가능함. => policy gradient를 써서 세타 C를 업데이트한다.


### Accelerate Training with Parallelism and Asynchronous Updates

- 자식 네트워크 : 하나의 모델을 뜻함
- 여러 컨트롤러 * 여러 자식 네트워크 => 많은 네트워크를 만들어냄
    - 훈련 속도를 높이기 위해 `파라미터-서버` 구조 사용

![3](https://miro.medium.com/max/700/1*9UQdtOqDpyef44nhKYsrcA.png)

S개의 파라미터 서버가 있고 이 서버와 연결된 K개의 복제된 **컨트롤러에 공유된 파라미터 값이 저장됨.**

각각의 컨트롤러는 m개의 자식 네트워크를 복제해서 **병렬**로 훈련시킴.

이 때 자식 네트워크의 정확도는 파라미터 서버에 보낼 세타 C에 대한 gradient를 계산하기 위해 기록됨.

3. Increase Architecture Complexity with Skip Connection and Other Layer Types


![스크린샷 2020-08-29 오후 12 14 03](https://user-images.githubusercontent.com/48315997/91627249-22e45180-e9f1-11ea-9774-73701c1e18c4.png)

Skip connection을 추가해서 탐생 공간을 넓힌다.

**레이어마다 anchor point를 더해서 이전 레이어들 중 어떤 레이어를 현재 레이어의 input으로 할지 결정함.**


4. Generate Recurrent Cell Architectures

지금까지 CNN을 위한 Neural Architecture, 지금은 RNN

![4](https://miro.medium.com/max/700/1*MFYHqx5BOad936u6QfhmrQ.png)

RNN, LSTM은 x(t), h(t-1)을 input으로 하고 h(t)를 output으로 하는 트리구조로 나타낼 수 있음(맨 오른쪽)

RNN 컨트롤러에서는 트리 노드들의 결합방석(addition, elementwise multiplication)과 활성화함수(sigmoid,tanh)를 선택할 수 있음.

> 그림 (b)의 Cell indices 의 왼쪽 1부분이 의미하는 것은 다음 메모리구조 C_t와 연결되는것은 Tree index 1 이며 오른쪽 0부분은 h_t 를 구할때 사용되는 것이 Tree index 0 이라는 것입니다. 그림 (b)의 Tree index 2 는 Tree0과 Tree1의 결합방식을 나타내는 것으로 그림에선 elementwise multiplication와 sigmoid의 결합이 됩니다.

## Experiments

기존 SOTA 모델과 비교했을 때 약간의 성능 감소는 있었지만 **더 작은 파라미터로 구현이 되었음,**

- CNN (CIFAR-10 dataset)

![스크린샷 2020-08-29 오후 12 22 27](https://user-images.githubusercontent.com/48315997/91627442-4fe53400-e9f2-11ea-98be-cb4432fd5b75.png)


- RNN (Penn Treebank dataset)

![스크린샷 2020-08-29 오후 12 23 03](https://user-images.githubusercontent.com/48315997/91627448-64c1c780-e9f2-11ea-8047-2596427ffe02.png)

- Transfer Learning on Neural Machine Translation
    - LSTM을 빼고 NAS를 통해 만든 cell을 넣었음.
    - LSTM에 특화된 하이퍼파라미터들을 튜닝하지 않음
    - BELU score 0.5 오름



## Additional studies
(If you have some parts that cannot understand, you have to do additional studies for them. It’s optional.)

- Understanding Deep Learning Requires Rethinking Generalization

- Designing Neural Network Architectures Using RL


## References
(References for your additional studies)

https://www.youtube.com/watch?v=XP3vyVrrt3Q

https://medium.com/@sunwoopark/slow-paper-neural-architecture-search-with-reinforcement-learning-6de601560522

