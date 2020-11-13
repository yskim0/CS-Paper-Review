![Page1](https://user-images.githubusercontent.com/49134038/99080548-5af33d80-25b9-11eb-9e96-a098224eb46b.png)   
![Page2](https://user-images.githubusercontent.com/49134038/99080561-60e91e80-25b9-11eb-8a57-6b3c3b045d42.png)   
![Page3](https://user-images.githubusercontent.com/49134038/99080563-621a4b80-25b9-11eb-9a8c-b76095e8d3ed.png)   
![Page4](https://user-images.githubusercontent.com/49134038/99080567-62b2e200-25b9-11eb-9067-0b3bf435a9cc.png)   
![Page5](https://user-images.githubusercontent.com/49134038/99080570-634b7880-25b9-11eb-9eba-b631e0d70602.png)   
   
## Code
### Discriminator & Generator
```
def get_fc_discriminator(num_classes, ndf=64):
    return nn.Sequential(
        nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=2, padding=1),
    )
```   
```
def get_deeplab_v2(num_classes=19, multi_level=True):
    model = ResNetMulti(Bottleneck, [3, 4, 23, 3], num_classes, multi_level)
    return model
```
```
class ResNetMulti(nn.Module):
    def __init__(self, block, layers, num_classes, multi_level):
        self.multi_level = multi_level
        self.inplanes = 64
        super(ResNetMulti, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)  # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        if self.multi_level:
            self.layer5 = ClassifierModule(1024, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)
        self.layer6 = ClassifierModule(2048, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if (stride != 1
                or self.inplanes != planes * block.expansion
                or dilation == 2
                or dilation == 4):
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine=affine_par))
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False
        layers = []
        layers.append(
            block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if self.multi_level:
            x1 = self.layer5(x)  # produce segmap 1
        else:
            x1 = None
        x2 = self.layer4(x)
        x2 = self.layer6(x2)  # produce segmap 2
        return x1, x2

    def get_1x_lr_params_no_scale(self):
        """
        This generator returns all the parameters of the net except for
        the last classification layer. Note that for each batchnorm layer,
        requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
        any batchnorm parameter
        """
        b = []

        b.append(self.conv1)
        b.append(self.bn1)
        b.append(self.layer1)
        b.append(self.layer2)
        b.append(self.layer3)
        b.append(self.layer4)

        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj += 1
                    if k.requires_grad:
                        yield k

    def get_10x_lr_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """
        b = []
        if self.multi_level:
            b.append(self.layer5.parameters())
        b.append(self.layer6.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i

    def optim_parameters(self, lr):
        return [{'params': self.get_1x_lr_params_no_scale(), 'lr': lr},
                {'params': self.get_10x_lr_params(), 'lr': 10 * lr}]
```   
### Domain Adaptation
```
```   
   
## Result
![1](https://github.com/feipan664/IntraDA/blob/65a118754b063285f2d93cc66e15b3bb4166328d/figure/introduction.png)   
![2](https://github.com/feipan664/IntraDA/blob/65a118754b063285f2d93cc66e15b3bb4166328d/figure/results.png)   
![3](https://github.com/feipan664/IntraDA/blob/65a118754b063285f2d93cc66e15b3bb4166328d/figure/examples.png)   
   
## Background
adversarial learning-based UDA approaches (Murez et al., 2017, Hoffman et al., 2017, Tsai et al., 2019, Tsai et al., 2018, Vu et al., 2018)   
the entropy of pixel-wise output predictions (Tsai et al., 2019)   
generating pseudo labels for target data and conducting refinement via an iterative self-training process.   
address the issue of multiple source domains; it focus on the multiple-source single-target adaptation setting.   
   
1.	Unsupervised Domain Adaptation.
2.	Uncertainty via Entropy.
3.	Curriculum Domain Adaptation.   
   
## Reference
Hoffman, J., Tzeng, E., Park, T., Zhu, J., Isola, P., Saenko, K., Efros, A.A. & Darrell, T. 2017, "CyCADA: Cycle-Consistent Adversarial Domain Adaptation", CoRR, vol. abs/1711.03213.   
Murez, Z., Kolouri, S., Kriegman, D.J., Ramamoorthi, R. & Kim, K. 2017, "Image to Image Translation for Domain Adaptation", CoRR, vol. abs/1712.00479.   
Tsai, Y., Hung, W., Schulter, S., Sohn, K., Yang, M. & Chandraker, M. 2018, "Learning to Adapt Structured Output Space for Semantic Segmentation", CoRR, vol. abs/1802.10349.   
Tsai, Y., Sohn, K., Schulter, S. & Chandraker, M. 2019, "Domain Adaptation for Structured Output via Discriminative Patch Representations", CoRR, vol. abs/1901.05427.   
Vu, T., Jain, H., Bucher, M., Cord, M. & P 'erez, P. 2018, "ADVENT: Adversarial Entropy Minimization for Domain Adaptation in Semantic Segmentation", CoRR, vol. abs/1811.12833.   
