# Reports for Deep Learning Society of undergraduates in Ewha

### Date: Aug. 22, 2020
### Student ID number / Name: 1785044 / 김연수
### Name of the thesis: 

[Meta Reinforcement Learning As Task Inference](https://arxiv.org/abs/1905.06424)


## Overview
(You should include contents of summary and introduction.)

- 접근 방법 : Meta-RL을 하나의 **paritally observed**로 본다. 
    - MDP의 모든 정보를 agent가 전부 받는 게 아니라 부분적으로만 관찰
    - RL은 하나의 control문제로 볼 수 있는데, 여기에 **inference problem**이 추가된 것

- 즉 POMDP(Partially-Observable Markov Decision Processes)문제를 해결하는 문제가 됨.
    - POMDP의 솔루션 : observation trajetory를 가지고 optimal policy를 찾는 것(미래 보상이 최대가 되는 쪽으로) == Reinforcement learning
        - 그러나 주어진 데이터가 partially observation이기 때문에 또 하나의 모듈이 필요함 -> `belief state`

- Belief State : 어떤 observation trajectory가 주어졌을 때 실제 true state의 probability
    - 이걸 구할 수 있으면 POMDP가 구해짐
        - POMDP가 구해지면 Meta-RL 문제 해결!

**Observation이 주어졌을 때 optimal policy를 구하는 것은 belief state를 구할 수 있으면 문제가 풀린다!**
-> 그러면 Meta-RL 문제도 풀린다.

- 이 논문의 key point : 두 가지 Neural Network를 사용
    - control하는 policy Network
    - belief state를 예측하는 inference Belief Module
        - auxilary supervised learning 로 해결 (Meta-learning때에만)
        - 즉, belief module은 meta-learning 때 supervised learning으로 학습됨.
        

- off-policy 사용(Meta-RL 에서는 대개 on-policy)


## Related work (Basic concepts)

- Meta Learning : Learning to Learn

![스크린샷 2020-08-22 오전 2 09 41](https://user-images.githubusercontent.com/48315997/90916720-8c29fa80-e41c-11ea-9484-b6129a1688d3.png)

- A `trajectory` is just a sequence of states and actions. 

- Meta RL
    - env.는 `MDP`로 표현됨. `M = {S,A,P,r}`
    - Agent <-> env. : 서로 interact하면서 future reward를 Max. 시키는 세타 찾기
    - Meta learning : 여러 개의 task들을 sampling하여 meta-learning한 후, meta-test시 빠르게 adaption될 수 있어야 함.
        - RL -> 그 task들 각각이 하나의 MDP로 정의 가능!

        ![스크린샷 2020-08-22 오전 2 13 34](https://user-images.githubusercontent.com/48315997/90917047-170af500-e41d-11ea-9791-74d540738406.png)

        - MDP를 여러개 sampling해서 학습하고, 그걸 통해 optimal theta얻는 것이 목표
        - test시에는 처음 보는 태스크(전체적인 쉐잎은 비슷해야)에 잘 적용되어야 함.

    - **접근 방법**
        1. Recurrent policies : RNN
        2. Optimization problem : MAML
        3. **partially observed RL**
            - MDP의 모든 정보를 받는 게 아니라 부분적으로만 받는다 -> inference problem 추가됨

            ![스크린샷 2020-08-22 오전 2 24 11](https://user-images.githubusercontent.com/48315997/90917816-93520800-e41e-11ea-81ba-5cd461f513e4.png)

            - z : task들의 모든 정보를 담고 있는 true information
            - z를 inference하면서 RL 컨트롤도 할 수 있는 해석하는 관점


- MDP(Markov Decision Processes) : (X,A,P,p0, R, discount factor)
![스크린샷 2020-08-22 오전 2 28 15](https://user-images.githubusercontent.com/48315997/90918158-2428e380-e41f-11ea-8e6b-8803cce60db7.png)
    - control이 있다 -> 대표적으로 RL
    - state가 완전히 관측되냐, 부분적으로 관측이 되냐 -> MDP/POMDP

- POMDP(Partially-Observable Markov Decision Processes)
    - MDP의 general한 버전
    - MDP에서 omega, O가 추가됨
    - X : state space. Agent 입장에서는 부분적으로 관측/아예 관측할 수 없게 됨.
    - 그래서 Agent는 부분적으로 관측되는 observation state를 통해서만 학습할 수 있음.
    ![스크린샷 2020-08-22 오전 2 30 23](https://user-images.githubusercontent.com/48315997/90918340-70742380-e41f-11ea-9513-7f6c21fb4280.png)


- off-policy algorithm : 현재 학습하는 policy가 과거에 했던 experience도 학습에 사용이 가능하고, 심지어는 해당 policy가 아니라 예를 들어 사람이 한 데이터로부터도 학습을 시킬 수가 있다.

- on-policy : 1번이라도 학습을 해서 policy improvement를 시킨 순간, 그 policy가 했던 과거의 experience들은 모두 사용이 불가능하다. 
    
## Methods
(Explain one of the methods that the thesis used.)

- Solution to POMDP
<img width="400" alt="스크린샷 2020-08-22 오전 2 35 32" src="https://user-images.githubusercontent.com/48315997/90918725-293a6280-e420-11ea-958a-b2c10565cbcc.png">

   - observed trajectory를 가지고 optimal policy를 찾는 것(미래 보상의 합이 최대가 되는 action set, policy) == RL
    - 그러나 주어진 데이터가 **partially** 하기 때문에 또 하나의 모듈이 필요함
    - **Belief State** 를 구할 수 있으면 위의 문제가 풀림.
    - Belief State -> POMDP -> Meta-RL 해결

- 최근 Meta-RL에서의 POMDP 해석 문제
    - 방금까지는 unobserved **state**
    - But 우리는 unobserved **task**
    ![스크린샷 2020-08-22 오전 2 40 44](https://user-images.githubusercontent.com/48315997/90919150-e331ce80-e420-11ea-8cc8-2c009ca75d8e.png)
   - 내가 어느 task를 풀고 있는지 모르는 상태
   - **state를 완전히 관측을 못한다는 것이 아니라!! 내가 어떤 task를 풀고 있는지를 관측하지 못한다는 관점으로 문제를 푸는 것임** == 어떤 task를 풀어야 하는지 task 정보를 완전하게 관측할 수 X -> 그래서 `Task Inference`

   <img width="400" alt="스크린샷 2020-08-22 오전 2 46 36" src="https://user-images.githubusercontent.com/48315997/90919603-b5995500-e421-11ea-9939-c4dc28820f91.png">

    - **State, Action은 모든 MDP간에 sharing되어 있음.** W가 붙은 것들은 task-specific함.
        - 이 세가지에 접근해서 Meta RL을 푼다.
    - 목표 : meta-test시 처음 보는 task에도 적은 interaction으로 reward를 Max. 시키는 optimal policy 찾기

- Meta-RL using POMDP
<img width="400" alt="스크린샷 2020-08-22 오전 2 50 42" src="https://user-images.githubusercontent.com/48315997/90919921-4839f400-e422-11ea-83ac-7e557be96989.png">

- A, S는 sharing
        - S는 true state와 Agent 입장에서 모르는 task에 대한 w를 concat해서 만듦
    - 나머지들은 task-specific하게 정의
    - w만이 agent입장에서는 unobserved state라고 정의

- optimal agent pi*는 아래와 같은 문제를 푼다.
<img width="347" alt="스크린샷 2020-08-22 오전 2 55 53" src="https://user-images.githubusercontent.com/48315997/90920278-01003300-e423-11ea-803a-2b74a4d6a2d7.png">

- agent입장에서는 실제 task label인 w에 access 할 수 없다고 가정. (task에 대한 MDP를 다 모르는 것)
    - 과거 observation trajectories은 LSTM, GRU등으로 agent network는 학습할 수 있음.
    - POMDP 문제를 해결하려면 task에 대한 belief state를 계산할 수 있어야 함.
        - observation trajectory가 주어졌을 때 실제 true task일 확률 (posterior)
            - 이걸 계산할 수 있으면 POMDP 문제 해결, 그러나 계산 어려움. 
        - appendix)
        <img width="707" alt="스크린샷 2020-08-22 오전 3 00 31" src="https://user-images.githubusercontent.com/48315997/90920660-a74c3880-e423-11ea-9b68-b972b3d4da7d.png">
        
- belief state의 posterior는 bayes Rule과 유사
- **policy가 식 안에 없음 -> task에 대한 posterior는 policy와 independent하다.**
- off-policy algorithm 사용 가능 (보통 meta-RL에서는 on-policy : 데이터를 모으자마자 바로 업데이트)

### Train

어떻게 모델을 학습시키는가

- current state `x`와 current belief `b_t(w)`만 있으면 가능
<img width="196" alt="스크린샷 2020-08-22 오전 3 03 34" src="https://user-images.githubusercontent.com/48315997/90920902-145fce00-e424-11ea-9bf4-9fc28d3c02f7.png">

- 앞서 언급한 z 가 여기서는 belief state(task에 대한 모든 정보를 담고 있다)
- belief state estimate 가능 -> optimal policy 구할 수 있다.
- 컴퓨팅 어려움 -> 2가지 NN 모델로 approximate 해야 한다.

1. Control하는 Policy Network

2. Belief Module

- 어떻게 학습시키는가?
    - Meta-Learning 안에서는 Supervised Learning이 된다.
        - label 가능성
        1. **Task Description : task를 잘 표현하는 representation.**
        2. Expert actions
        3. Task embeddings
    <img width="500" alt="스크린샷 2020-08-22 오전 3 10 40" src="https://user-images.githubusercontent.com/48315997/90921367-124a3f00-e425-11ea-8784-c78d7e116bd5.png">
    - **meta-learning 때에만 사용.** 실제 우리가 원하는 meta-test에서 적은 에피소드만으로 빠르게 adapt할 때는 이 정보들을 더이상 사용하지 않음.
    - task를 supervised learning으로 풀었다!
    - w, h_t : independent of policy => belief network도 off-policy로 효율적으로 학습 가능

### Architecture

<img width="736" alt="스크린샷 2020-08-22 오전 3 25 18" src="https://user-images.githubusercontent.com/48315997/90922528-1d05d380-e427-11ea-8e3a-9fe4b1417e21.png">

LSTM, IB : optional (IB : regularization 효과)

A. Baseline LSTM Agent : belief network가 없는 일반 agent

B. 논문에서 제안된 모델 : trajectory 넣어서 **Belief network 학습**
    - Belief feature + trajectory 해서 **policy network 학습**
    - 각자 역할에 집중

C. Mixed : Network 하나에 합친 것


### Experiment

<img width="755" alt="스크린샷 2020-08-22 오전 3 28 30" src="https://user-images.githubusercontent.com/48315997/90922806-900f4a00-e427-11ea-8b7e-6c06f607daf5.png">

- Multi-armed bandit : 20 arms and 100 horizon. 
    - 20번 당겼을 때 reward probability => task description. (arm들의 vec.)

- Semicircle : 반원 안에서 naviagtion
    - task desription : 각도

<img width="736" alt="스크린샷 2020-08-22 오전 3 33 40" src="https://user-images.githubusercontent.com/48315997/90923240-483cf280-e428-11ea-9280-d01cc4fba786.png">

- cheetah : task description -> velocity

<img width="634" alt="스크린샷 2020-08-22 오전 3 40 23" src="https://user-images.githubusercontent.com/48315997/90923769-3871de00-e429-11ea-893d-b52b23dcbb83.png">


### Main contributions

1. Supervised Learning을 통해서 performance 높임
2. Belief network 학습시킬 때 off-policy 알고리즘 사용 가능
3. continuous, sparse rewards 환경에서도 좋았다


## Additional studies
(If you have some parts that cannot understand, you have to do additional studies for them. It’s optional.)

POMDP 

Belief State?



## References
(References for your additional studies)

https://www.youtube.com/watch?v=phi7_QIhfJ4 - 논문 설명

https://newsight.tistory.com/250 - policy 부분 개념
