# Image Reconstruction (2019)


## Introduction

#### 1. Imitation Phrase:
copy drawing = initializing networks   
#### 2. Generating Phrase:
preliminary painting = reconstruct preliminary images   
#### 3. Refinement Phrase:
fine-tuned = utilize to fine-tune preliminary images into final detailed outputs   


## Background

#### 1. Sketch-to-image (S2I) synthesis   
- Sketch Based Image Retrieval (SBIR) [1]   
 + +edgel2 index: index structure (shape image를 document-like representation으로: image to word vector), structure-consistent sketch: database image와 sketch query(image)의 유사도 측정 structure   
 + -추출된 features가 많아 매칭 시 문제 발생

- Cross-modal retrieval   
  * Instance-level [2]   
  + +Sketchy database: shared embedding for sketches and photos 학습 (distances in the learned feature space = sketches and photos의 구조/의미 유사도)   
  
  * Category-level [3]   
  + +the missing modes problem 최소화, autoencoder-based regularizers로 training 안정화   
  
  + -pixel level image에 적용하기 힘듦
  + -style을 input으로 적용하기 힘듦 (SBIR의 self-limitation of retrieval 특성: 특정 위치, 각도 등 too specific 매칭)

- Scribbler [4]

#### 2. Image-to-image (I2I) translation   
- Pix2Pix [9]   
- CycleGAN [6]   
- multi-modal I2I translation tasks [7]   
- BicycleGAN [8]   
- Unsupervised multi-modal I2I translation methods [10]   


## Model
   
### PI-REC
* sparser edges and pixel-level color style 을 input으로 받아 with both high-fidelity in content and style image를 만들어냄.   
* Imitation Phase, Generating Phase and Refinement Phase 로 구성되어 있으며 하나의 Generator와 Discrriminator 이용.   

#### 1. Preprocessing of training Data
+ Edge   
  - Canny algorithm 이용: rough but solid binary edges
+ Color Domain   
  - median filter algorithm   
  - K-means algorithm: color domain 평균값 추출   
  - median filter (AGAIN!!): sharpness of the boundary lines를 blurry하게 만듦.   
+ Hyperparameter Confusion (HC)   
  - 특정 범위 내의 different random values of hyperparameters 적용: training datasets를 증가시켜 overfitting 방지   
  - extracted edge의 each pixel은 8%의 확률로 0 값으로 reset 될 수 있음 (스케치의 선이 edge일 확률을 92%로 가정한 것.): generalization ability 향상   

#### 2. Model Architecture
![overall](https://github.com/youyuge34/PI-REC/blob/master/files/architecture_v5.png)   

+ Generator      
 ``` Imitation Phase ```   
 optimality 도달 = output distribution p(X_fake-1)이 the distribution of ground truth image p(Xgt)가 같은 상태 (the Imitation Phase output = X_fake-1)   
 ``` Generating Phase ```   
 - G1-1이 (0으로) 수렴 = Generator가 inirialized features를 잘 학습한 상태.   
 - detail을 더 빠르게 생성, 수렴한 결과 빨리 도출 (with input: edge E and color domain C_gt)   
 ``` Refinement Phase ```   
 - G1-3을 의미
 - checkerboard artifact 줄이고 more high frequency details 생성 및 optimize the color distribution

+ Discriminator   
PatchGAN architecture with spectral normalization 이용: fake를 검출하기 위해 larger receptive field 사용

#### 3. Model loss
* Per-pixel Loss: the loss difference L1 between X_fake and X_gt   
* Adversarial loss: LSGAN (a stable generator 생성: distribution of real images with high frequency details에 최적화)   
* Feature loss: perceptual losses (the differences in style 극대화)   
* Style loss: perceptual losses, 위와 동일   


## Result
![1](https://github.com/youyuge34/PI-REC/blob/master/files/s_banner4.jpg)   
``` APPENDIX) 기타 DEMO: https://github.com/youyuge34/PI-REC/ 참고 ```


## References
1. Edgel Index for Large-Scale Sketch-based Image Search
2. The Sketchy Database: Learning to Retrieve Badly Drawn Bunnies
3. MODE REGULARIZED GENERATIVE ADVERSARIAL NETWORKS
4. Scribbler: Controlling Deep Image Synthesis with Sketch and Color
5. SketchyGAN: Towards Diverse and Realistic Sketch to Image Synthesis
6. Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks
7. ComboGAN: Unrestrained Scalability for Image Domain Translation
8. High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs
9. Diverse Image-to-Image Translation via Disentangled Representations
10. Toward multimodal imageto-image translation
