# Playing Atari with Deep Reinforcement Learning

## Overview
- High-Dimensional Sensory Input으로부터 Reinforcement Learning을 통해 Control Policy를 성공적으로 학습하는 Deep Learning Model을 선보인다.
- 이 모델에서 Atari는 CNN 모델을 사용하며, 변형된 Q-learning을 사용하여 학습된다.
- Q-learning이란 input이 raw pixels이고, output은 미래의 보상을 예측하는 value function
- 실제로 게임을 학습할 때, 스크린의 픽셀값들을 입력으로 받고, 각 행위에 대해 점수를 부여하고, 어떤 행동에 대한 결과값을 함수를 통해 받게 된다.
- Atari는 2600개가 넘는 다양한 게임을 학습시키는데 동일한 모델과 학습 알고리즘을 사용한다.

> Background
### < Q-learning > : Model이 없이(Model-Free) 학습하는 강화 학습 알고리즘
- Q-Value

Q-Learning에서는 어떤 State S에서 어떤 Action A를 했을 때, 그 행동이 가지는 Value를 계산하는 Q-Value를 사용한다. (행동-가치 함수)
알고리즘은 각 상태-행동 쌍에 대해 Q:S×A→R.Q:S×A→R.같은 Q-Function을 갖는다.
- Q-Learning Algorithm


### < Experience Replay Memory >

전체 데이터의 분포를 보면 a가 정답에 가장 근접한 직선이지만, b근처의 데이터만으로 학습을 진행한다고 하면 b가 정답에 가장 가까운 직선이 된다.

> Whole Process; Agent가 환경(Atari Emulator) 와 상호작용하는 task
1. 각 time-step마다 Agent는 할 수 있는 행동( at) 들 중에서 한가지를 선택
2. Action이 전달되면 Emulator는 내부 상태를 변경하고 게임 점수를 수정 
3. 이를 해결하기 위해 action의 sequence를 관찰하고 이를 통해 학습을 진행 (이러한 Formalism은 크지만 유한한 Markov Decision Process(MDP)을 야기하는데, 여기서 각 시퀀스는 별개의 상태에 해당)
4. 결과적으로 MDP에 standard한 reinforcement learning method를 적용할 수 있고, 이것은 시간 t에서의 상태를 표현하기 위해 전체 시퀀스를 사용함을 의미
5. Agent의 목표는 Future Reward을 극대화시키는 방식으로 action을 선택하고 이를 Emulator에
전달하는 것인데, 시간이 오래 지날수록 그 reward의 가치는 점점 내려가므로 이를 적용시키기 위해 discount factor r이 정의된다.  

## Related work


## Deep Reinforcement Learning 
Experience replay memory라는 기술을 활용하였는데, Agent가 매 Time Step마다 했던 Experience (Episode)들을 Dataset에 저장을 시키고, 수많은 Episode들이 replay memory에 쌓이게 된다. 그리고 알고리즘 내부에서 샘플들이 저장된 풀로부터 임의로 하나를 샘플링하여 학습(Q_Learning, Mini_Batch)에 적용시켰다. 이후에(experience replay 후) Agent는 e_greedy policy에 따라 행동을 선택하고 수행한다. Neural Network의 입력으로써 가변적인 history를 사용하는 것은 어렵지만, Deep_Q Algorithm에서는 ϕ함수를 사용하여 고정 길이의 history를 입력으로 사용한다.
 
> DQN
DQN에서는 replay memory 안에 마지막 N개의 experience만을 저장하고, update를 하기위해 무작위로 Data Set으로부터 추출한다. 이러한 접근법은 Memory Buffer가 중요한 Transition에 차별점을 두지 않고 항상 제한된 크기 N의 버퍼에 최근의 Transition을 덮어 씌운다는 점에서 한계가 있다. 마찬가지로, uniform sampling은 replay memory안의 모든 transition에 동일한 중요성을 부여한다. 더욱 정교한 Sophisticated Sampling 전략은 우선순위를 매기는 것과 유사하게 우리에게 가장 중요한 Transition을 중요시 할 것이다.



### 4.1 Preprocessing and Model Architecture
< Preprocessing >
128  color palette를 가진 210 * 160 pixel images인 raw Atari frames로 직접 작업하는 것은 정말 많은 계산양을 필요로 한다. 그래서 먼저 input의 dimensionality를 줄이는 basic preprocessing step을 적용한다. 이러한 과정은 먼저 RGB로 표현된 이미지를 Gray-Scale로 변환하고, 110 * 84의 이미지로
down-sampling 시킨다. 
이후에 게임의 진행 부분만 보이도록 84 * 84로 잘라내서 Final Input값을 추출한다. (사용하는 GPU에서 정사각형 사진만 GPU 연산이 가능하기 때문이다.) 위의 알고리즘의 전처리 함수인 ϕ()에서 마지막 4개의 frames만 전처리를 하여 Stack에 넣어두고, 이를 입력에 대한 Q_function의 값을 구하기 위해 사용한다.

### < Model Architecture >
Value를 구하는 방법으로는 2가지가 있다.
1.	history와 action을 input으로 하고, output으로 history와 그 action에 대해 예측된 Q-value를 구하는 것
2.	history만을 input으로 하여 output으로 각 행동에 대해 예측된 Q-Value를 구하는 것 

### < DNN Architecture >

1. Neural Network의 Input은 ϕ 를 통해 전처리된 84 x 84 x 4 이미지(4 frames)이다.
2. 첫 번째 Hidden Layer는 input image에 stride 4를 토함한 16 8x8(16 channels with 8x8 filters)로 합성곱 연산을 한 후에, rectifier non-linearity(ex. relu)를 적용한 것이다.
3. 두번째 Hidden Layer는 stride 2를 포함한 32 4x4(32 channels with 4x4 filters)로 합성곱 연산을 하고 rectifier non-linearity(ex. relu)를 적용한 것이다.
4. 마지막 Hidden Layer는 funnly-connected되고, 256개의 rectifier 유닛으로 구성된다.
5. 최종적으로 Output layer는 각 수행가능한 행동에 대해 single output을 갖는 fully-connected linear layer이다.

## Experiments

왼쪽의 두 그래프는 Sequest, Breakout이라는 게임에서 학습을 하는동안 total_reward가 어떻게 변화하는지 보여준다. 


screen의 왼쪽에 enemy가 등장하였을 때, predicted value가 jump함을 보여준다(Point A). 
적을 발견한 Agent는 Enemy를 향해 미사일을 발사하고, 발사된 미사일이 적을 맞추려고 할 때, predicted value가 오름을 보여준다(Point B). 
그리고 적이 사라졌을 때 predicted value는 원래의 값으로 떨어지게 된다(Point C). 

## Code


## Additional studies -
### off-policy

##References
https://mangkyu.tistory.com/60
