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
  +Sketchy database: shared embedding for sketches and photos 학습 (distances in the learned feature space = sketches and photos의 구조/의미 유사도)   
  
  * Category-level [3]   
  +the missing modes problem 최소화, autoencoder-based regularizers로 training 안정화   
     
  -pixel level image에 적용하기 힘듦
  -style을 input으로 적용하기 힘듦 (SBIR의 self-limitation of retrieval 특성: 특정 위치, 각도 등 too specific 매칭)

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

``` Generator ```      
 1. Imitation Phase   
 optimality 도달 = output distribution p(X_fake-1)이 the distribution of ground truth image p(Xgt)가 같은 상태 (the Imitation Phase output = X_fake-1)   
 2. Generating Phase   
   - G1-1이 (0으로) 수렴 = Generator가 inirialized features를 잘 학습한 상태.   
   - detail을 더 빠르게 생성, 수렴한 결과 빨리 도출 (with input: edge E and color domain C_gt)   
 3. Refinement Phase   
   - G1-3을 의미
   - checkerboard artifact 줄이고 more high frequency details 생성 및 optimize the color distribution
   
``` Discriminator ```    
PatchGAN architecture with spectral normalization 이용: fake를 검출하기 위해 larger receptive field 사용

#### 3. Model loss
* Per-pixel Loss: the loss difference L1 between X_fake and X_gt   
* Adversarial loss: LSGAN (a stable generator 생성: distribution of real images with high frequency details에 최적화)   
* Feature loss: perceptual losses (the differences in style 극대화)   
* Style loss: perceptual losses, 위와 동일   

   
## CODE

```
def test_G(self): // test
        self.g_model.eval()

        // create_dir(self.results_path)

        test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
        )

        index = 0
        for items in test_loader:
            name = self.test_dataset.load_name(index)
            images, images_gray, edges, color_domain = self.cuda(*items)
            // # print('images size is {}, \n edges size is {}, \n color_domain size is {}'.format(images.size(), edges.size(), color_domain.size()))
            index += 1

            outputs = self.g_model(edges, color_domain)
            outputs = output_align(images, outputs)
            outputs_merged = outputs

            output = self.postprocess(outputs_merged)[0]
            // path = os.path.join(self.results_path, name)
            // print(index, name)

            // imsave(output, path)

            // 
            if self.debug:
                images_input = self.postprocess(images)[0]
                edges = self.postprocess(edges)[0]
                color_domain = self.postprocess(color_domain)[0]
                fname, fext = name.split('.')
                fext = 'png'
                imsave(images_input, os.path.join(self.results_path, fname + '_input.' + fext))
                imsave(edges, os.path.join(self.results_path, fname + '_edge.' + fext))
                imsave(color_domain, os.path.join(self.results_path, fname + '_color_domain.' + fext))
            //

       // print('\nEnd test....')
```
   
```
def test_R(self): // refinement
        self.r_model.eval()

        create_dir(self.refine_path)

        test_loader = DataLoader(
            dataset=self.refine_dataset,
            batch_size=1,
        )

        index = 0
        for items in test_loader:
            name = self.refine_dataset.load_name(index)
            images, images_gray, edges, _ = self.cuda(*items)
            # print('images size is {}, \n edges size is {}, \n color_domain size is {}'.format(images.size(), edges.size(), color_domain.size()))
            index += 1

            outputs = self.r_model(edges, images)
            outputs = output_align(images, outputs)
            outputs_merged = outputs

            output = self.postprocess(outputs_merged)[0]
            path = os.path.join(self.refine_path, name)
            print(index, name)

            imsave(output, path)

            if self.debug:
                images_input = self.postprocess(images)[0]
                edges = self.postprocess(edges)[0]
                # color_domain = self.postprocess(color_domain)[0]
                fname, fext = name.split('.')
                fext = 'png'
                imsave(images_input, os.path.join(self.refine_path, fname + '_input.' + fext))
                imsave(edges, os.path.join(self.refine_path, fname + '_edge.' + fext))
                # imsave(color_domain, os.path.join(self.results_path, fname + '_color_domain.' + fext))

        print('\nEnd refinement....')
```
   
```
def test_G_R(self): // test with refinement // overall process
        self.g_model.eval()
        self.r_model.eval()

        create_dir(self.results_path)

        test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
        )

        index = 0
        for items in test_loader:
            name = self.test_dataset.load_name(index)
            images, images_gray, edges, color_domain = self.cuda(*items)
            # print('images size is {}, \n edges size is {}, \n color_domain size is {}'.format(images.size(), edges.size(), color_domain.size()))
            index += 1

            outputs = self.g_model(edges, color_domain)
            outputs = output_align(images, outputs)
            outputs_merged = outputs

            output = self.postprocess(outputs_merged)[0]
            path = os.path.join(self.results_path, name)
            print(index, name)

            imsave(output, path)

            if self.debug:
                images_input = self.postprocess(images)[0]
                edge = self.postprocess(edges)[0]
                color_domain = self.postprocess(color_domain)[0]
                fname, fext = name.split('.')
                fext = 'png'
                imsave(images_input, os.path.join(self.results_path, fname + '_input.' + fext))
                imsave(edge, os.path.join(self.results_path, fname + '_edge.' + fext))
                imsave(color_domain, os.path.join(self.results_path, fname + '_color_domain.' + fext))

            img_blur = outputs
            # img_blur = self.cuda(img_blur)
            outputs = self.r_model(edges, img_blur)

            output = self.postprocess(outputs)[0]
            # output = outputs.cpu().numpy().astype(np.uint8).squeeze()
            fname, fext = name.split('.')
            fext = 'png'
            imsave(output, os.path.join(self.results_path, fname + '_refine.' + fext))

        print('\nEnd test with refinement....')
```
   
```
def draw(self, color_domain, edge): // edge
        self.g_model.eval()
        size = self.config.INPUT_SIZE
        color_domain = resize(color_domain, size, size, interp='lanczos')
        edge = resize(edge, size, size, interp='lanczos')
        edge[edge <= 69] = 0
        edge[edge > 69] = 255

        color_domain = to_tensor(color_domain)
        edge = to_tensor(edge)

        color_domain, edge = self.cuda(color_domain, edge)

        if self.config.DEBUG:
            print('In model.draw():---> \n color domain size is {}, edges size is {}'.format(color_domain.size(),
                                                                                             edge.size()))
        outputs = self.g_model(edge, color_domain)

        outputs = self.postprocess(outputs)[0]
        output = outputs.cpu().numpy().astype(np.uint8).squeeze()
        edge = self.postprocess(edge)[0]
        edge = edge.cpu().numpy().astype(np.uint8).squeeze()

        return output
```
   
```
def refine(self, img_blur, edge): // blurry
        self.r_model.eval()
        size = self.config.INPUT_SIZE
        # color_domain = resize(color_domain, size, size, interp='lanczos')
        edge = resize(edge, size, size, interp='lanczos')
        edge[edge <= 69] = 0
        edge[edge > 69] = 255

        img_blur = to_tensor(img_blur)
        edge = to_tensor(edge)

        img_blur, edge = self.cuda(img_blur, edge)

        if self.config.DEBUG:
            print('In model.refine():---> \n img_blur size is {}, edges size is {}'.format(img_blur.size(),
                                                                                             edge.size()))
        outputs = self.r_model(edge, img_blur)

        outputs = self.postprocess(outputs)[0]
        output = outputs.cpu().numpy().astype(np.uint8).squeeze()
        edge = self.postprocess(edge)[0]
        edge = edge.cpu().numpy().astype(np.uint8).squeeze()

        return output
```
   
```
def postprocess(self, img):
        # [0, 1] => [0, 255]
        img = img * 255.0
        img = img.permute(0, 2, 3, 1)   
        
        > the 4 dimensions of input_patch are <batch size, image height, image width, image channel> respectively.   
        > In Pytorch, the input channel should be in the second dimension. That's why the permutation is required.   
        > After the permutation, the 4 dimensions of in_img will be <batch size, image channel, image height, image width>.   
        
        return img.int()
```

   
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
