# A3C: Asynchronous Methods for Deep Reinforcement Learning

## Introduction

### 기존 DQN의 문제점
_DQN은 Playing Atari with Deep Reinforcement Learning 논문 Review 참고_   

> [DQN]   
> online RL algorithm과 deep neural network의 조합   
> (online: RL agent가 학습 환경에서 시간 순으로 데이터를 얻음)   
1. non-stationary problem when policy updates
2. correlation of observed data   
  + 연속적인 경험은 비슷하므로 관계를 정의하기 때문   
<br>
이를 해결하기 위해서 experience replay memory에 agent data 저장   
-> random sampling으로 replay memory에서 batch를 꺼내 학습   

  - off-policy 형태로 강제됨   

### multiple agents in multiple indepedent instance environment in parallel
**Parallelism** : decorrelating agents' data => on-policy & off-policy 작동 가능   
: 여러 agent를 동시(주어진 time-step)에 다른 environment에서 Action하여 Experience 쌓기 -> 결과 학습 네트워크 공유   
  * __multi-threaded__: stationary policy를 갖게 됨   
  
~~Experience Replay Memory 사용하지 않음~~

## Related Work

1. Gorila (General Reinforcement Learning Architecture)   
: distributed setting에서 agent들이 asynchronous training 수행   
  * 각 agent가 각각의 environment의 experience를 replay memory에 저장
  > 해당 replay memory에서 random sampling하여 DQN loss를 통해 policy 학습   
  > -> DQN loss의 parameter로 미분된 gradient를 _central parameter server_로 전달
  > -> model의 global environment update
  > -> update된 model의 policy parameter은 fixed interval을 통해 agent에게 전달됨   
2. Map Reduce Framework   
: linear function approximation을 통해 batch RL algorithm을 parallel 학습
  * Parallelism: large matrix operation 가속화, agent의 experience를 parallel 수집하는 것이 아님.   

3. parallel Sarsa Algorithm   
: multiple agents -> training 가속화
  * 각 actor는 독립적으로 학습 수행, peer-to-peer communication을 이용해 주기적으로 학습된 weight 전달하여 퍼지게 함.   
  
## Reinforcement Learning Background
_Playing Atari with Deep Reinforcement Learning 논문 Review 참고_   
> 에이전트가 판단하는 방식을 정책(Policy)이라고 부르는데, 에이전트와 혼용하는 경우가 많습니다. 정책을 수학적으로 나타내면 상태에 따른 행동의 조건부 확률, 즉 P(action|state)가 됩니다.

## Asynchronous RL Framework

### Asynchronous 1-step Q-Learning   
![1_Q](https://user-images.githubusercontent.com/40893452/45151193-9be8ae00-b208-11e8-9f65-7b9717e7fbcf.png)   

### Asynchronous 1-step Sarsa   
Asynchronous 1-step Q-learning에서 다른 target value Q(s,a)를 사용한다는 것 빼고 모두 동일

### Asynchronous n-step Q-Learning   
![n_Q](https://user-images.githubusercontent.com/40893452/45205494-46220d80-b2bd-11e8-8445-76374c9a5830.png)   
> forward view(== n step 앞의 결과)를 본다는 점에서 일반적이지 않습니다. 이런 forward view를 사용하는 것은 neural network를 학습하는 과정에서 momentum-based methods와 backpropagation 과정에서 훨씬 더 효과적인 학습이 가능하도록 해 줍니다. 한번의 업데이트를 위해서, 알고리즘은 policy를 기반으로 action을 고르며 최대 t(max)-step(또는 state가 끝날 때 까지)까지 미리 action을 고릅니다. 이 과정을 통해 agent가 t(max)까지의 rewards를 마지막으로 update했던 state으로부터 한번에 받아옵니다.

### Asynchronous Advantagge Actor-Critic(A3C) RL   
![overall](http://openresearch.ai/uploads/default/original/1X/710e633cbbad482b3b424e5d95162a2039995778.jpg)   
: 여러 agent가 각기 다른 environment에서 Action을 취하며 Experience를 얻음
  + 1 machine에서 multi-thread 구현 (agent의 최대 개수 == thread 개수)   
  + 각 agent가 각기 다른 exploration policy를 갖음 => 성능 및 Robustness 증가   
: 실제로 expected value보다 얼마나 더 나았는지 *Advantage*를 계산 -> loss에 사용    
  ```
  Advantage_A = Q(s,a) - V(s)   
  estimatedAdvantage_A' = R - V(s)
  ```   
  > Q-Learning: discounted return을 직접 estimate   
<br>

#### Actor-Critic
Actor: policy를 통해 action을 취하는 Agent   
Critic: value function을 통해 현재 상태를 Evaluate

#### A3C Algorithm
* n-step Q-learning 알고리즘와 같이 forward view를 사용해서 policy와 value function을 업데이트   
* policy와 value function 들은 모두 t(max) or terminal state에 도착한 후에 업데이트   

![update](https://user-images.githubusercontent.com/40893452/45300004-3407cf80-b548-11e8-847a-70cfd5fb3e6e.png)   
<br>

* * *

##### 모든 agent들이 공유하는 네트워크의 output *(모든 non-output layer들의 가중치는 공유)*
  1. policy π(At|St;θ): policy는 π (at | st; θ)에 대해 하나의 softmax 출력을 가지는 convolutional neural network를 사용  
  
  > policy π의 엔트로피를 loss function에 더하면 suboptimal 로의 premature convergence를 방지하여 exploration을 개선한다는 것을 발견했다.   
  
  ![loss](https://user-images.githubusercontent.com/40893452/45300917-982b9300-b54a-11e8-8422-ad89709e1d88.png)   
  
  2. value function, V(St;θv): value-function V(st; θv)에 대해 하나의 선형 출력을 가짐   
  
  > θ,θv는 분리되어 있는 parameter가 아닌, 공유되는 parameter이다. (일부 parameter는 세상에서 공유됨)   
  
* * *

##### Loss
1. Policy Loss: Lp = log(π(s)) * A(s)   
  _Advantage (A(s))가_   
  - 양수 == 기대보다 좋은 경우: policy의 action이 1의 방향으로 training   
  - 음수 == 기대보다 안 좋은 경우: policy의 action이 0의 방향으로 training   
  => Advantage(실제 값 - 예측 값)을 줄어드는 방향으로 training == 실제 값과 예측 값이 비슷해지는 방향   
  **이렇게 action이 얼마나 좋아져야 하는지 판단하기 때문에 "Action-Critic"이라고 부름**   
  > 논문에서는 위의 일반적인 loss에 π에 대한 entropy loss를 아래와 같이 추가해서, 보수적인 모델로의 early converge를 막는다고 합니다.   
  > ![policy_loss](https://user-images.githubusercontent.com/40893452/45300917-982b9300-b54a-11e8-8422-ad89709e1d88.png)   
<br>
2. Value Loss: W = sum(R - V(s))_2   
```
__최종 loss__
L = Lp' + 0.5 * Lv
```   

#### Overall Flow
![flow](http://openresearch.ai/uploads/default/original/1X/aa019a73a51f4a5e5d7db25d7e5c06e336be20d6.jpg)   
1. thread 별로 생성된 agent가 shared parameter로부터 동일한 network(== global network) copy   
2. 각 agent는 서로 다른 environment에서 서로 다른 exploration policy를 가지고 exploration   
3~5. 각 agent가 value / policy loss로부터 gradient를 구하고 asynchronous하게 global network에 전달하여 shared parameter 업데이트   
![code](http://openresearch.ai/uploads/default/original/1X/ecab76979198a73a645eb2c739797a9889e210c8.jpg)   

## Experiments
1.   
![Fig.1](http://openresearch.ai/uploads/default/original/1X/8ba0f1317daaa02b6e7c201edf89bab6001c79f2.png)   
![Fig.2](http://openresearch.ai/uploads/default/original/1X/95ec528070bbb67b374e856440eb087e6f17a69e.png)   
Atari 2600에서 대표적인 5가지 게임을 선정했고, DQN에 비해 더 짧은 시간에 더 높은 퍼포먼스를 보이는 agent를 안정적으로 학습했음

2.   
![Fig.3](http://openresearch.ai/uploads/default/original/1X/03439146388612e42d9fee1d8b679d06fd4f0dbb.png)   
thread 수에 따라 매우 효과적으로 scale-up 됨   
  - Data Exploration Ability의 향상으로 parallel한 worker 수가 증가함에 따라 트레이닝 시간이 단축됨   
  > 놀랍게도 선형 이상으로 증가하기도 하는데, 이는 적은 수일때에 대비해서 bias를 제거해주는 effect로 해석했습니다.   
  
3.   
![Fig.4](http://openresearch.ai/uploads/default/original/1X/db91be85605c914e28ec82267cb4c97ef13b4590.png)   
Robust & Stable   
  - RL에서 트레이닝 되던 Agent가 Collapse하거나 Diverge 하는 경우가 있어 트레이닝이 상대적으로 어려운데, A3C의 경우 hyperparameter 등에 따라 robust함   
  - 특히 트레이닝이 잘되는 Learning Rate 구간 값 내에서는 Collapse 되는 결과 없었음

## Code
https://github.com/awjuliani/DeepRL-Agents/blob/master/A3C-Doom.ipynb   

## Reference
http://openresearch.ai/t/a3c-asynchronous-methods-for-deep-reinforcement-learning/25   
https://github.com/170928/-Review-Asynchronous-Methods-for-Deep-Reinforcement-Learning   
https://medium.com/@awjuliani/super-simple-reinforcement-learning-tutorial-part-1-fd544fab149   
https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-8-asynchronous-actor-critic-agents-a3c-c88f72a5e9f2
