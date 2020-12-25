블로그 포스팅 -> https://yskim0.github.io/paper%20review/2020/12/25/DCGAN/

-----

# Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks (DCGAN)

GAN은 Representation Learning에 효과적이라고 합니다. (본 논문에서는 이것이 GAN의 learning process와 관련이 있고, lack of heuristic cost function 덕분이라고 말합니다.)


그럼에도 불구하고 처음 나온 GAN은 학습하는 것에 있어 매우 **unstable**하다는 문제점을 가지고 있었습니다.


본 논문의 `DCGAN`은 해당 문제점 해결을 포함하여 총 4가지의 Contributions로 정리할 수 있습니다.

```
1. DCGAN은 stable training이 가능하다
2. 학습된 Discriminator는 image classification 태스크에 사용이 가능하다. (다른 unsupervised 알고리즘들과 비교할 것임.)
3. GAN이 학습한 filters를 시각화할 수 있고, 특정 오브젝트에 대해 특정 filter를 학습했다는 점을 보여줄 수 있다.
4. Generator는 vector arithmetic properties를 가지고 있다.
```



## APPROACH AND MODEL ARCHITECTURE

<img width="705" alt="스크린샷 2020-12-25 오후 5 31 03" src="https://user-images.githubusercontent.com/48315997/103127996-f73f5280-46d6-11eb-9936-996a9ae0e84f.png">


우선 DCGAN이 기존의 GAN과 architecture 측면에서 `어떻게` 달랐기에 큰 성과를 이룰 수 있었는지 살펴보겠습니다. 


DCGAN이 나오기 전까지에도 CNN을 이용한 GAN을 만드는 시도는 계속 있었습니다. 하지만 이 시도들은 모두 성공적이지 못했죠. 


DCGAN은 **CNN Architecture에서의 최신 변화(?) 3가지를 적용**하여 성공하였습니다.



### All Convolutional Net 사용 

우선 가장 큰 변화점은 [Striving for Simplicity: The All Convolutional Net](https://arxiv.org/abs/1412.6806) 을 참고하여 All Convoultional Net을 사용했다는 점입니다.


All conv. net은 **pooling functions(예를 들어, max pooling)를 strided conv. 으로 바꾼 네트워크 구조**입니다. Max pooling은 미분되지 않는 성질을 가진다고 합니다.

이 네트워크 구조를 통해 Generator와 Discriminator 모두 자신들의 spatial downsampling을 학습하기에 적합해집니다.


### Eliminating fully connected layers(FC layers) on top of convolutional features

이 시점 trend는 마지막에 FC layer를 제거하고, `global average pooling`를 쓰는 방식이었다고 합니다. 


다만, 이 global average pooling은 **모델의 안정성은 올리는 반면에 convergence speed는 떨어뜨리는** trade-off 관계를 가지고 있습니다.


### Batch Norm & ReLU activation

**Batch Normalization**는 학습을 안정화시킬 수 있는 방법 중 하나입니다. (normalizing the input to each unit to have zero mean and unit variance)


이것은 학습 문제점 중 하나인 initialization과 deep model의 gradient flow에 큰 도움을 줄 수 있습니다. 


**하지만, 모든 layer에 BN을 적용시키는 것은 아닙니다.** Generator의 output layer와 Discriminator의 input layer에는 BN을 적용시키지 않습니다.


마지막으로 **ReLU를 사용했다**는 점이 언급되어 있는데, Generator와 Discriminator에 적용되는 function이 약간 다릅니다. 

Generator에는 기본적으로 ReLU를 사용하지만, output layer에는 따로 ReLU가 아닌 Tanh function을 적용시킵니다.


Discriminator에는 Leaky ReLU를 적용합니다.


(여러 실험 결과 이 functions들이 좋은 성능을 냈기 때문이겠죠? 확실한 이유, 논증은 모르겠습니다.)



## DETAILS OF ADVERSARIAL TRAINING


DCGAN은 `LSUN, FACES, IMAGENET-1K` 데이터셋에 대하여 학습하였습니다.


training setting(parameters, ...)에 대해서는 논문을 참고해주시고, 위 데이터셋 중 LSUN에 대해서만 알아보겠습니다 :) 


### LSUN

LSUN은 *Large-scale Scene Understanding*의 줄임말로, bedroom 사진들을 모은 데이터 셋입니다.


이 데이터셋을 가지고 학습시킨 모델로 생성된 이미지는 quality가 상당히 향상되었습니다. 


하지만 이것이 **over-fitting이 되어 이렇게 된 것인지, 학습 데이터셋에 대하여 기억(memorization)하여 만들어진 것인지**에 대해 판별해봐야 합니다.


- overfitting?

<img width="700" alt="스크린샷 2020-12-25 오후 5 45 38" src="https://user-images.githubusercontent.com/48315997/103128506-00c9ba00-46d9-11eb-8b7a-e144208bbfe8.png">


본 논문의 Fig.2 와 Fig.3을 통해 모델이 오히려 **underfitting** 되어있음을 얘기합니다.

underfitting이 이루어졌다고 여기는 이유는 아직 noise texture가 눈에 보이기 때문이라고 합니다.

*(엄청 가까이 보지 않는 이상은 잘 느끼지 못하겠는데 말이죠.)*


- memorization?

사실 이게 가장 중요한 부분이라고 볼 수 있습니다. 새로 generate 하는 image가 사실 학습 데이터에서 기억(memorize)하여 만들어진 것이라면, **진정한 의미의 Generate가 아니니깐요.**


이에 대해서 본 논문의 저자들은 memorize하는 가능성을 줄이기 위해 `image de-duplication process`(중복 제거 프로세스)를 거칩니다.


de-duplication을 하기 위해 autoencoder를 하나 만듭니다. 

> We fit a 3072-128-3072 de-noising dropout regularized RELU autoencoder on 32x32 downsampled center-crops of training examples.




## EMPIRICAL VALIDATION OF DCGANs CAPABILITIES

맨 처음에 이 논문의 contributions 중 하나로 `학습된 Discriminator는 image classification 태스크에 사용이 가능하다` 라고 얘기했었죠?


이 목차에서는 정말 DCGAN이 **feature extractor**로써의 역할이 가능한가, 그래서 CIFAR-10 데이터셋에 대해서도 **classification task**를 잘 수행하는가를 확인해봅니다.


<img width="696" alt="스크린샷 2020-12-25 오후 6 03 36" src="https://user-images.githubusercontent.com/48315997/103129277-8484a600-46db-11eb-9e0f-de26d03694d4.png">


결론적으로, 다른 unsupervised 알고리즘은 K-means model들보다 우수한 성능을 보였습니다. (Exemplar CNN 모델보다는 좀 못 미치지만요.)


<img width="542" alt="스크린샷 2020-12-25 오후 6 05 55" src="https://user-images.githubusercontent.com/48315997/103129364-d6c5c700-46db-11eb-8f7b-388b5ae3128b.png">


추가적으로 SVHN Digits 데이터셋에 대해서도 실험을 해보았습니다. test error 측면에서 SOTA를 달성하는 쾌거를 이루었습니다.



## INVESTIGATING AND VISUALIZING THE INTERNALS Of THE NETWORKS



또 다른 contributions으로  `GAN이 학습한 filters를 시각화할 수 있고, 특정 오브젝트에 대해 특정 filter를 학습했다는 점을 보여줄 수 있다` 와 `Generator는 vector arithmetic properties를 가지고 있다` 가 있었습니다. 


이 목차에서는 두 부분에 대해 설명할 수 있습니다.



### Walking in the Latent Space


latent space를 변경했을 때 sharp transitions(급작스러운 변화)가 있으면 이는 `memorization`이 일어났다는 신호일 수 있습니다. 


반대로 부드러운 변화가 일어나면 memorization이 된 것이 아니라 제대로 학습이 되었다고 볼 수 있죠.


![image](https://user-images.githubusercontent.com/48315997/103134579-83607280-46f5-11eb-800b-b4d5ea21bdad.png)


위의 사진을 보시면 DCGAN의 경우 sharp transition이 아닌 smooth한 변화가 이루어졌음을 볼 수 있습니다.



### Visualizing the Discriminator Features


이 내용에서는 Guided backpropagation을 통해 GAN이 학습한 filters를 시각화할 수 있습니다. 


<img width="696" alt="스크린샷 2020-12-25 오후 9 17 38" src="https://user-images.githubusercontent.com/48315997/103134734-9f184880-46f6-11eb-9639-d0eb9878e74a.png">


discriminator가 feature들을 학습해서, 특정 파트(bed, windows,...)들에 대하여 active 하고 있음을 볼 수 있습니다.


### Manipulating the Generator Representation


#### Forgetting to Draw Certain Objects


이건 매우 재미있는 실험입니다.


간단하게 요약하자면, Generator가 무슨 representation을 학습했는지 알아보기 위하여 특정 filter(여기서는 window filter)를 삭제해봅니다. 


즉, Window라는 object를 `Forget` 하게 되는 것이죠.(기술적으로는 window filter를 dropout 시킨다고 말할 수 있습니다.) 


결과적으로 이 실험에서는 창문이 아닌 다른 representations, objects가 들어가게 됩니다!


<img width="684" alt="스크린샷 2020-12-25 오후 9 22 03" src="https://user-images.githubusercontent.com/48315997/103134811-3da4a980-46f7-11eb-8d95-b990a0a18601.png">


위의 결과를 보면 아시다시피, 창문이었던 것이 문으로 바뀌는 등 다른 object를 생성합니다.



#### Vector Arithmetic On Face Samples


많은 분들이 가장 재밌어하실(?) 부분인 것 같습니다.



word embedding 관련해서 vector("King") - vector("Man") + vector("Woman")가 vector("Queen")의 결과가 나오듯이, DCGAN에서도 이와 비슷한 **arithmetic한 연산이 가능하다**고 밝혔습니다.


Generator의 input인 Z vector에 대한 arithmetic operation을 하는데, single sample로는 불안정하여 3개의 Z vector를 평균한 값을 사용한다고 합니다. 


<img width="580" alt="스크린샷 2020-12-25 오후 9 36 16" src="https://user-images.githubusercontent.com/48315997/103135086-3a122200-46f9-11eb-931d-7d737e8c5ed1.png">


smiling woman - neutral woman + neutral man = smiling man 이미지가 만들어지는 마법같은 기술을 보실 수 있습니다.

앞서 말했다시피 3개의 Z vector를 average하여 새로운 Y벡터를 만든 것도 확인해볼 수 있죠.


이게 다가 아닙니다. **face pose** 또한 Z space에 선형적으로 모델링할 수 있습니다.

<img width="557" alt="스크린샷 2020-12-25 오후 9 38 01" src="https://user-images.githubusercontent.com/48315997/103135120-780f4600-46f9-11eb-8d7e-8d723c10e439.png">


바로 이렇게 말이죠. 
이미 이전부터 scale, rotation, position에 대하여 conditional generative model은 학습할 수 있다고 연구되어왔습니다. 하지만 이 연구는 `purely unsupervised model` 이라는 점에서 큰 변환점이 된 것이죠.


## FUTURE WORK


사실 stablity를 완전히 해결한 것은 아닙니다. 


DCGAN을 오랫동안 학습하게 되면 collapse mode, oscillating mode가 발생할 수 있습니다. 아직도 **불안정성**이 남은 것이죠. 


그래서 이 논문에서는 해당 문제점을 Future work로 남기고 마무리하였습니다.



---


DCGAN도 읽었으니, 구현도 해보고 후속 논문들도 찬찬히 읽어보려고 합니다.
긴 글 읽어주셔서 감사합니다 :)
