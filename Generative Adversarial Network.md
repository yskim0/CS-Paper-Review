# Generative Adversarial Network

## Overview
‘Generative’: ‘분포’를 만드는 모델을 학습하는 것이 GAN의 목적이다.
‘Adversarial’: 지폐 위조범에게 Generator라는 역할을 부여하자. 그리고 경찰은 Discriminator라는 역할을 부여한다. 각각의 역할을 가진 두 개의 모델을 통해 진짜같은 가짜를 생성해내는 능력을 키워주는 것이 GAN의 핵심 아이디어이다. 
 즉, D의 목표는 Real, 혹은 Fake 이미지를 제대로 분류해내는 것이다. 그리고 G의 임무는 완벽하게 D가 틀리도록 하는 것이다. 그래서 두 코어 모델의 Loss 지표는 반대가 되며, 이 때문에도 '적대적' 모델로 불린다.

- 먼저 이미지를 판별하는 Discriminator(이하 D)는 CNN (판별기, classifier)과 같이 구성할 수 있다. 
- G는 random한 noise를 생성해내는 vector z를 input으로 하며(그림의 Noise), D가 판별하고자 하는 input image(위 그림의 28X28 mnist 이미지)를 output으로 하는 neural network unit이라고 할 수 있다.

> 학습 과정
1. 실제 mnist 이미지(Real Image)를 D로 하여금 ‘진짜’라고 학습시킨다.
2. 이후, vector z와 G에 의해 생성된 Fake Image를 ‘가짜’라고 학습시킨다.
* D가 2번 학습되고 G는 1번 학습되는 것이 아니라,
  1.에서의 Real Image와 2.의 Fake Image를 D의 x input으로 합쳐서 총 1번 학습한다는 것이다. 

## Code
```
def train_D(self):
        """
        train Discriminator
        """

        # Real data
        real = self.data.get_real_sample()

        # Generated data
        z = self.data.get_z_sample(self.batch_size)
        generated_images = self.gan.G.predict(z)

        # labeling and concat generated, real images
        x = np.concatenate((real, generated_images), axis=0)
        y = [0.9] * self.batch_size + [0] * self.batch_size

        # train discriminator
        self.gan.D.trainable = True
        loss = self.gan.D.train_on_batch(x, y)
        return loss

    def train_G(self):
        """
        train Generator
        """

        # Generated data
        z = self.data.get_z_sample(self.batch_size)

        # labeling
        y = [1] * self.batch_size

        # train generator
        self.gan.D.trainable = False
        loss = self.gan.GD.train_on_batch(z, y)
        return loss
```
- train_D는 D를 학습하는 부분, 그리고 train_G는 D(G(z))에서 G를 학습하는 부분이다.
- D.trainable을 사용하여, 위에서 설명한 대로 D는 한 번만 학습되도록 구현하였다. 따라서, (train_G의) D(G(z))에서 D의 학습을 False로 한다면 G만 학습이 된다. D를 학습시키기 위해서는
‘x = np.concatenate((real, generated_images), axis=0)’을 통해 진짜 이미지와 가짜 이미지를 concatenate하여 한번에 학습시킨다.

## Related work (Basic concepts)
< Adversarial Nets >
성공적으로 분류한 경우 = 1, 잘못 / 가짜를 진짜로 분류한 경우 = 0

(1). D가 (z를) 잘 구분한다면: D(x) = 1
(2). G가 (z를) 잘 생성한다면: D(G(z)) = 0 (D가 G(z)을 잘 구분하지 못함)
* 따라서 adversarial nets의 최대값(최상의 상태)은 0이다.

## Theoretical Results

(a). 와 가 부분적으로 비슷하므로 D는 부분적으로 정확한 분류자
(b). inner loop를 통해 D가 data 및 sample을 구별하도록 학습한 경우 (D : G의 학습 비율은 1 : 5 와 같은 형태로 불균형하게 하는 것이 일반적인 듯 하다. D가 G에 비해 너무 정확하다면, G의 gradient가 vanishing되는 문제가 생기기도 하고, 반대의 경우도 생긴다.)
(c). G(z)가 training data와 유사해지도록 G를 update한 후 D가 G(z)를 잘 분류할 수 있도록 gradient 유도
(d). G(z)가 training data와 같아지는, 한 global optimal 상태 도달, D는 1:1의 비율(=1/2의 확률)로 구분 가능

### 4.1 Global Optimality of p_g = p_data

* 목적은 maximized Discriminator의 기능을 minimizing하는 Generator 생성
G가 fixed된 상태에서 arbitrary G에 대한 optimal discriminator D 구하기: maximizing the quantity V(G,D)

C(G)가 G가 minimum이 되는 optimal discriminator이라고 가정하면, 

p_g = p_data는 D(x) = 1/2을 의미하므로,

(C(G)에서 D(x)=y일 때 미분 적용)

### 4.2 Convergence of Algorithm1

multilayer perceptron인 adversarial nets가 항상 optimal neural network를 내놓지 않기 때문에 global optimal 상태를 가정한 위의 과정이 항상 맞지는 않을 것이라 예상했으나, 실제로는 이러한 multilayer perceptron network가 reasonable한 model을 생성함.

## Code
### DCGAN
[ Generator ]
```
import parser  class Generator(nn.Module):     def __init__(self):         super(Generator, self).__init__()          self.init_size = opt.img_size // 4         self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size ** 2))          self.conv_blocks = nn.Sequential(             nn.BatchNorm2d(128),             nn.Upsample(scale_factor=2),             nn.Conv2d(128, 128, 3, stride=1, padding=1),             nn.BatchNorm2d(128, 0.8),             nn.LeakyReLU(0.2, inplace=True),             nn.Upsample(scale_factor=2),             nn.Conv2d(128, 64, 3, stride=1, padding=1),             nn.BatchNorm2d(64, 0.8),             nn.LeakyReLU(0.2, inplace=True),             nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),             nn.Tanh(),         )      def forward(self, z):         out = self.l1(z)         out = out.view(out.shape[0], 128, self.init_size, self.init_size)         img = self.conv_blocks(out)         return img
```

[ Discriminator ]
```
import parser  class Discriminator(nn.Module):     def __init__(self):         super(Discriminator, self).__init__()          def discriminator_block(in_filters, out_filters, bn=True):             block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]             if bn:                 block.append(nn.BatchNorm2d(out_filters, 0.8))             return block          self.model = nn.Sequential(             *discriminator_block(opt.channels, 16, bn=False),             *discriminator_block(16, 32),             *discriminator_block(32, 64),             *discriminator_block(64, 128),         )          # The height and width of downsampled image         ds_size = opt.img_size // 2 ** 4         self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())      def forward(self, img):         out = self.model(img)         out = out.view(out.shape[0], -1)         validity = self.adv_layer(out)          return validity
```

[ Training ]
```
import parser  Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor  # ---------- #  Training # ----------  for epoch in range(opt.n_epochs):     for i, (imgs, _) in enumerate(dataloader):          # Adversarial ground truths         valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)         fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)          # Configure input         real_imgs = Variable(imgs.type(Tensor))          # -----------------         #  Train Generator         # -----------------          optimizer_G.zero_grad()          # Sample noise as generator input         z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))          # Generate a batch of images         gen_imgs = generator(z)          # Loss measures generator's ability to fool the discriminator         g_loss = adversarial_loss(discriminator(gen_imgs), valid)          g_loss.backward()         optimizer_G.step()          # ---------------------         #  Train Discriminator         # ---------------------          optimizer_D.zero_grad()          # Measure discriminator's ability to classify real from generated samples         real_loss = adversarial_loss(discriminator(real_imgs), valid)         fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)         d_loss = (real_loss + fake_loss) / 2          d_loss.backward()         optimizer_D.step()          print(             "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"             % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())         )          batches_done = epoch * len(dataloader) + i         if batches_done % opt.sample_interval == 0:             save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
```

## Additional studies
1. KL (Kullback-Leibler divergence)

2. JSD (Jensen-Shannon divergence)


## References
https://leedakyeong.tistory.com/entry/논문Generative-Adversarial-NetsGAN
https://www.youtube.com/watch?v=L3hz57whyNw&list=PLlMkM4tgfjnJhhd4wn5aj8fVTYJwIpWkS&index=2
https://subinium.github.io/VanillaGAN/
https://yamalab.tistory.com/98
