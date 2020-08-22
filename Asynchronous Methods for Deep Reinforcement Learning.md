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
이를 해결하기 위해서 **experience replay memory** 에 agent data 저장   
-> random sampling으로 replay memory에서 batch를 꺼내 학습   
  - off-policy 형태로 강제됨   

### multiple agents in multiple indepedent instance environment in parallel
**Parallelism** : decorrelating agents' data => on-policy & off-policy 작동 가능   

## Related Work

1. Gorila (General Reinforcement Learning Architecture)   
: distributed setting에서 agent들이 asynchronous training 수행   
  * 각 agent가 각각의 environment와 독립적인 replay memory를 가짐
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


## Asynchronous RL Framework
  
  
  
  
  
  
