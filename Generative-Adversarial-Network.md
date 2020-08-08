# Reports for Deep Learning Society of undergraduates in Ewha

### Date: Aug. 8, 2020
### Student ID number / Name: 1785044 / 김연수
### Name of the thesis: 

[Generative Adversarial Nets](https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf)


## Overview
(You should include contents of summary and introduction.)


GAN에는 두 가지 모델이 존재함.
- Discriminator
- Generator

<br>

**Image를 만들어 내는 Generator(G)가 이 만들어진 모델을 평가하는 Discriminator(D)를 최대한 속일 수 있도록, 확률 분포의 차이를 줄이는 것이 목적**

- 즉, G는 D를 최대한 속이려고 노력하고, D는 G가 만든 이미지를 최대한 감별하려고 노력함.
- 이 경쟁 속에서 두 모델은 모두 발전하게 되고, 결과적으로는 G가 만든 이미지를 구별할 수 없는 상태에 도달하게 됨. 

<img width="592" alt="스크린샷 2020-08-08 오후 12 18 01" src="https://user-images.githubusercontent.com/48315997/89702019-b5a44a00-d977-11ea-971b-f3e800d5ac7b.png">

<br>



위의 목표를 이루기 위해서는, *(ref. output -> [0,1] : 0==false, 1==true)*
- D : 진짜 이미지를 진짜 이미지라고 인식(분류)하도록 학습
- G : random한 코드를 받아서 img를 생성한 후, 그 이미지가 D를 속여야 함.
    - 즉, D(G(z)) = 1(진짜라 인식)이 나오도록 학습.
        - 학습할수록 진짜같은 가짜 img가 만들어지는 것

## Related work (Basic concepts)

- generative model
- Adversarial 
- VAE



## Methods
(Explain one of the methods that the thesis used.)

### GAN loss/objective function

<img width="476" alt="스크린샷 2020-08-08 오후 12 27 25" src="https://user-images.githubusercontent.com/48315997/89702021-b806a400-d977-11ea-87ad-6f12b78a0939.png">

- D 입장에서는 위 수식이 0인게 Maximize
- G 입장에서는 속이는 게 좋으니 Mininmize



+) G는 처음에 형편없는 이미지를 만듦.
- D는 그 이미지를 가짜라 확신. => D(G(z))=0
- 하지만 위의 `log(1-x)` 로는 그때 기울기의 절댓값이 작음.
- practical use : D가 가짜라 확신하는 상황을 최대한 빨리 벗어나려면, D(G(z))=0인 점에서 기울기가 거의 무한인 `log(x)`를 씀

![스크린샷 2020-08-08 오후 12 33 14](https://user-images.githubusercontent.com/48315997/89702023-b937d100-d977-11ea-97d7-1245192f4975.png)


- 모델이 생성한 이미지 분포와 실제 이미지 분포 간의 차이를 계산해주는 함수로 `Jenson-Shannon divergence` 사용함.

### Approach

1. The minimax problem of GAN has a global opt. at p(g) = p(data)

- Proposition 1. 

![스크린샷 2020-08-08 오후 12 43 28](https://user-images.githubusercontent.com/48315997/89702024-bb9a2b00-d977-11ea-99ab-a850c92cf5f6.png)

![스크린샷 2020-08-08 오후 12 51 39](https://user-images.githubusercontent.com/48315997/89702026-bd63ee80-d977-11ea-9d89-6f2b99c6b7f4.png)

- Main Theorem.

위를 이용해서 D가 optimal 가정.

```
The global minimum of the virtual training criterion C(G) is achieved if and only if p(g)=p(data). 
At that point, C(G) achieves the value −log(4).
```

![IMG_51B0698FF18C-1](https://user-images.githubusercontent.com/48315997/89702260-6b709800-d97a-11ea-9553-fc2ccaddfea2.jpeg)






2. The proposed algorithm can find that global opt.

- 그래서 알고리즘이 위 문제를 풀 수 있는가를 확인


![스크린샷 2020-08-08 오후 12 57 04](https://user-images.githubusercontent.com/48315997/89702027-bf2db200-d977-11ea-85e9-4a4a6685a0da.png)


**1번==>minimax problem -> global opt. 가진다는 증명이었음.**
- global opt. -> 모델의 분포 == 실제 분포
- 즉 우리가 풀려는 문제 C(G)가 convex문제임을 확인했음.
    - minimization problem이 쉬워짐.

**MLP로 충분히 가능하다.**



### Vector arithmetic 하다.

안경 낀 남자 - 안경 안 낀 남자 + 안경 안 낀 여자 = 안경 낀 여자

## Code

```py
# https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/generative_adversarial_network/main.py

import os
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
latent_size = 64
hidden_size = 256
image_size = 784
num_epochs = 200
batch_size = 100
sample_dir = 'samples'

# Create a directory if not exists
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

# Image processing
# transform = transforms.Compose([
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean=(0.5, 0.5, 0.5),   # 3 for RGB channels
#                                      std=(0.5, 0.5, 0.5))])
transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5],   # 1 for greyscale channels
                                     std=[0.5])])

# MNIST dataset
mnist = torchvision.datasets.MNIST(root='../../data/',
                                   train=True,
                                   transform=transform,
                                   download=True)

# Data loader
data_loader = torch.utils.data.DataLoader(dataset=mnist,
                                          batch_size=batch_size, 
                                          shuffle=True)

# Discriminator
D = nn.Sequential(
    nn.Linear(image_size, hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size, hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size, 1),
    nn.Sigmoid())

# Generator 
G = nn.Sequential(
    nn.Linear(latent_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, image_size),
    nn.Tanh())

# Device setting
D = D.to(device)
G = G.to(device)

# Binary cross entropy loss and optimizer
criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002)

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

def reset_grad():
    d_optimizer.zero_grad()
    g_optimizer.zero_grad()

# Start training
total_step = len(data_loader)
for epoch in range(num_epochs):
    for i, (images, _) in enumerate(data_loader):
        images = images.reshape(batch_size, -1).to(device)
        
        # Create the labels which are later used as input for the BCE loss
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # ================================================================== #
        #                      Train the discriminator                       #
        # ================================================================== #

        # Compute BCE_Loss using real images where BCE_Loss(x, y): - y * log(D(x)) - (1-y) * log(1 - D(x))
        # Second term of the loss is always zero since real_labels == 1
        outputs = D(images)
        d_loss_real = criterion(outputs, real_labels)
        real_score = outputs
        
        # Compute BCELoss using fake images
        # First term of the loss is always zero since fake_labels == 0
        z = torch.randn(batch_size, latent_size).to(device)
        fake_images = G(z)
        outputs = D(fake_images)
        d_loss_fake = criterion(outputs, fake_labels)
        fake_score = outputs
        
        # Backprop and optimize
        d_loss = d_loss_real + d_loss_fake
        reset_grad()
        d_loss.backward()
        d_optimizer.step()
        
        # ================================================================== #
        #                        Train the generator                         #
        # ================================================================== #

        # Compute loss with fake images
        z = torch.randn(batch_size, latent_size).to(device)
        fake_images = G(z)
        outputs = D(fake_images)
        
        # We train G to maximize log(D(G(z)) instead of minimizing log(1-D(G(z)))
        # For the reason, see the last paragraph of section 3. https://arxiv.org/pdf/1406.2661.pdf
        g_loss = criterion(outputs, real_labels)
        
        # Backprop and optimize
        reset_grad()
        g_loss.backward()
        g_optimizer.step()
        
        if (i+1) % 200 == 0:
            print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}' 
                  .format(epoch, num_epochs, i+1, total_step, d_loss.item(), g_loss.item(), 
                          real_score.mean().item(), fake_score.mean().item()))
    
    # Save real images
    if (epoch+1) == 1:
        images = images.reshape(images.size(0), 1, 28, 28)
        save_image(denorm(images), os.path.join(sample_dir, 'real_images.png'))
    
    # Save sampled images
    fake_images = fake_images.reshape(fake_images.size(0), 1, 28, 28)
    save_image(denorm(fake_images), os.path.join(sample_dir, 'fake_images-{}.png'.format(epoch+1)))

# Save the model checkpoints 
torch.save(G.state_dict(), 'G.ckpt')
torch.save(D.state_dict(), 'D.ckpt')
```

## Additional studies
(If you have some parts that cannot understand, you have to do additional studies for them. It’s optional.)

이후 GAN 논문들


## References
(References for your additional studies)

https://www.youtube.com/watch?v=L3hz57whyNw


https://www.youtube.com/watch?v=odpjk7_tGY0


http://jaejunyoo.blogspot.com/2017/01/generative-adversarial-nets-2.html
