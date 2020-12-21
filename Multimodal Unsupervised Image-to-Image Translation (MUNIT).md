# Multimodal Unsupervised Image-to-Image Translation (MUNIT)

## Keywords : GAN, Image-to-Image translation, style transfer

- image representation = content code + style code
    - content code : domain invariant
    - style code : domain specific

<img width="657" alt="스크린샷 2020-12-22 오전 12 14 37" src="https://user-images.githubusercontent.com/48315997/102791653-adaaeb00-43ea-11eb-9dd6-fc7d4e6c6e6e.png">


- Image Translation == Recombine its content code with a random style code
    - `random style code` : style space of the target domain

<img width="651" alt="스크린샷 2020-12-22 오전 12 14 23" src="https://user-images.githubusercontent.com/48315997/102791642-a5eb4680-43ea-11eb-8ea5-8af9ebc97f17.png">

### Auto-encoder Architecture

<img width="624" alt="스크린샷 2020-12-22 오전 12 19 33" src="https://user-images.githubusercontent.com/48315997/102792116-61ac7600-43eb-11eb-8a7b-3f4cc735c0f1.png">
