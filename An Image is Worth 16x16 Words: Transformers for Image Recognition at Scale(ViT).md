# Reports for Deep Learning Society of undergraduates in Ewha

### Date: Oct. 31, 2020
### Student ID number / Name: 1785044 / 김연수
### Name of the thesis: 

[An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)

## Overview

- image classification task에 기존의 transformer 모델을 이용한다.
    - transformer의 장점들을 사용할 수 있음.
    - simple, computational efficiency, scalability

- Vision Transformer(ViT)
    - 원본 이미지를 **patches**들로 split한다.
        - 이 때 이미지 패치들을 NLP에서의 **token(word)**와 같은 역할임.
    - 이 패치들의 sequence of linear embedding을 Transformer의 input으로 feed한다.

- 크지않은 데이터 셋에서는 ResNet보다 약간 낮은 정확도
    - Transformer는 CNN과 다르게 *translation equivariance, locality* 같은 **inductive biases**(=weight sharing)이 없기 때문에 데이터셋이 적으면 generalize되기 어려움.
- 하지만 large scale 데이터에서는 CNN의 inductive bias를 능가함. 
    - image recognition benchmarks에서 여러 SOTA 달성함.


## Related work (Basic concepts)



## Methods
(Explain one of the methods that the thesis used.)

- Model overview
    <img width="777" alt="스크린샷 2020-11-01 오전 11 51 34" src="https://user-images.githubusercontent.com/48315997/97794098-97797d80-1c38-11eb-8221-30ea0e5d6ea7.png">

- Vision Transformer(ViT)
    - image를 patch 단위로 잘라서 flatten시킨 후, 그것을 linear projection하여 encoder에 feed한다.
    ![image](https://user-images.githubusercontent.com/48315997/97794316-8da54980-1c3b-11eb-963b-2c07da280861.png)

    - Equation
    <img width="737" alt="스크린샷 2020-11-01 오후 12 32 37" src="https://user-images.githubusercontent.com/48315997/97794501-55ebd100-1c3e-11eb-97cd-cc00bde85b4d.png">

    - (Eq.1) trainable linear projection은 flatten된 patch들을 *D* dimension에 **mapping시킨다.** 
    - (Eq.4) BERT의 `[class]` 토큰처럼, embedded patch의 sequence(image representation `y`) 전에 **learnable embedding**을 추가한다. (prepend)
        - both during pre-training and fine-tuning, a classification head is attached to \left({z}_{L}^{0} \right) 
    - Position embeddings
        - positional information을 유지하기 위해 patch embedding에 붙여짐. (자세한건 Appendix.D.3)
        - standard learnable 1D position embedding 사용
        - 이렇게 position embedding까지 더해진 sequence of embedding vector들은 encoder의 input이 됨.

    - Encoder
        - alternating layers of multiheaded self-attention(MSA)
        - MLP blocks
            - contains 2 layer with a GELU non-linearity
            ```py
            class MlpBlock(nn.Module):
            """Transformer MLP / feed-forward block."""

            def apply(self,
                        inputs,
                        mlp_dim,
                        dtype=jnp.float32,
                        out_dim=None,
                        dropout_rate=0.1,
                        deterministic=True,
                        kernel_init=nn.initializers.xavier_uniform(),
                        bias_init=nn.initializers.normal(stddev=1e-6)):
                """Applies Transformer MlpBlock module."""
                actual_out_dim = inputs.shape[-1] if out_dim is None else out_dim
                x = nn.Dense(
                    inputs,
                    mlp_dim,
                    dtype=dtype,
                    kernel_init=kernel_init,
                    bias_init=bias_init)
                x = nn.gelu(x)
                x = nn.dropout(x, rate=dropout_rate, deterministic=deterministic)
                output = nn.Dense(
                    x,
                    actual_out_dim,
                    dtype=dtype,
                    kernel_init=kernel_init,
                    bias_init=bias_init)
                output = nn.dropout(output, rate=dropout_rate, deterministic=deterministic)
                return output
            ```
        - Layernorm(LN) is applied before every block
        - residual connections after every block

    - Hybrid Architecture
        - patch embedding projection E (Eq.1)이 CNN feature map으로 대체될 수 있다.
            - 즉, ResNet과 같은 CNN구조의 모델을 가지고, 2D feature map 중 하나를 1D로 flatten시킨 후 transformer dimension에 projecting 시킴.
            - 위에서 만들어진 sequence에 classification input embedding, position embedding를 추가시켜 encoder에 input으로써 feed 시킬 수 있음.

### Fine-Tuning and Higher Resolution

- large dataset으로 pre-training하고, smaller downstream task에 대해 fine-tune 하려고 함.
- 이를 위해서 pre-trained prediction head를 지우고, 0으로 initializedgks D x K feedforward layer를 추가함.
    - K : # of downstream classes
- pre-training 보다 높은 resolution으로 fine-tuning하는 것은 beneficial할 때도 있음.
- higher resolution을 feed하게 되면, patch size는 동일하므로 sequence length가 길어짐
- ViT는 임의적인 sequence length를 다룰 수 있음(메모리 제약에 따라서)
- **하지만 pre-trained position embedding이 의미 없어질 수 있음.**
    - 원본 이미지의 location에 따라 pre-trained position embedding의 2D interpolation을 수행함.
- 위와 같은 resolution adjustment와 patch extraction은 이미지의 2D 구조에 대해 inductive bias를 **manually** ViT에 주입시키는 유일한 포인트임.

## Additional studies
(If you have some parts that cannot understand, you have to do additional studies for them. It’s optional.)



## Code

- `model.py`
    - [Github](https://github.com/google-research/vision_transformer/blob/master/vit_jax/models.py)

## References
(References for your additional studies)

https://jeonsworld.github.io/vision/vit/
