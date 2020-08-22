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
이를 해결하기 위해서 
**experience replay memory**
에 agent data 저장   
-> random sampling으로 replay memory에서 batch를 꺼내 학습   
  - off-policy 형태로 강제됨   

### multiple agents in multiple indepedent instance environment in parallel
**Parallelism** : decorrelating agents' data => on-policy & off-policy 작동 가능   
: 여러 agent를 동시(주어진 time-step)에 다른 environment에서 Action하여 Experience 쌓기 -> 결과 학습 네트워크 공유   
  * __multithreaded__: stationary policy를 갖게 됨
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
  '''
  Advantage_A = Q(s,a) - V(s)   
  estimatedAdvantage_A' = R - V(s)
  '''   
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
##### 모든 agent들이 공유하는 네트워크의 output *(모든 non-output layer들의 가중치는 공유)*
  1. policy π(At|St;θ): policy는 π (at | st; θ)에 대해 하나의 softmax 출력을 가지는 convolutional neural network를 사용  
  > policy π의 엔트로피를 loss function에 더하면 suboptimal 로의 premature convergence를 방지하여 exploration을 개선한다는 것을 발견했다.   
  ![loss](https://user-images.githubusercontent.com/40893452/45300917-982b9300-b54a-11e8-8422-ad89709e1d88.png)   
  2. value function, V(St;θv): value-function V(st; θv)에 대해 하나의 선형 출력을 가짐   
  > θ,θv는 분리되어 있는 parameter가 아닌, 공유되는 parameter이다. (일부 parameter는 세상에서 공유됨)

##### Loss


  
