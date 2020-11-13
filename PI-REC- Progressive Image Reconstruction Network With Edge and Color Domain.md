# Image Reconstruction (2019)

## Introduction
1. Imitation Phrase: copy drawing = initializing networks   
2. Generating Phrase: preliminary painting = reconstruct preliminary images   
3. Refinement Phrase: fine-tuned = utilize to fine-tune preliminary images into final detailed outputs   

## Background
1. Sketch-to-image (S2I) synthesis   
- Sketch Based Image Retrieval (SBIR) [1]   



- Cross-modal retrieval   
  * Instance-level [2]   
  
  * Category-level [3]   

- Scribbler [4]


2. Image-to-image (I2I) translation   
- Pix2Pix [9]   
- CycleGAN [6]   
- multi-modal I2I translation tasks [7]   
- BicycleGAN [8]   
- Unsupervised multi-modal I2I translation methods [10]   

## Model
![overall](https://github.com/youyuge34/PI-REC/blob/master/files/architecture_v5.png)   
   
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

#### 3. Model loss



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
