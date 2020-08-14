# Reports for Deep Learning Society of undergraduates in Ewha

### Date: Aug. 14, 2020
### Student ID number / Name: 1785044 / 김연수
### Name of the thesis: 

[Model-Agnostic Meta-Learning for fast adaptation of deep networks](https://arxiv.org/abs/1703.03400)


## Overview
(You should include contents of summary and introduction.)

좋은 Weight로 initialize하는 방법에 대한 것.
**어떻게 하면 좋은 initial weight를 찾을 수 있는가?**
**어떤 inital weight를 가지면 모르는 태스크들에 대해서도 빨리 적응시킬 수 있는가?**


- Key Point

<사진>
    - **주어진 태스크들에 대해서 1 step 갔을 때, 모든 태스크에 대해서 로스가 미니멈이 되는 현재의 세타를 찾는 것!**

## Related work (Basic concepts)

- Meta learning
    - learning to learn
    - 좋은 메타 러닝 모델 = 트레이닝 때 접하지 않았던 새로운 태스크나 환경에 대해서 잘 적응되거나 일반화가 잘 됨.
    - Reinforcement learning과 결합한 **meta-learning**(meta reinforcement learning) 얘기가 많이 나오고 있음
    - Few-shot classification은 supervised-learning 상황에서 meta-learning을 활용한 예시임.
        - **하나의 데이터셋 자체가 하나의 data sample로 활용되고 있음.**
        <사진>
        - 즉 Meta-learning에서는 training, test의 개념이 일반과 약간 다르고, 그 때 들어가는 데이터셋도 다르다.
        - 약간의 fine-tuning 과 유사한 접근법



## Methods
(Explain one of the methods that the thesis used.)

<사진>




## Code


## Additional studies
(If you have some parts that cannot understand, you have to do additional studies for them. It’s optional.)



## References
(References for your additional studies)
