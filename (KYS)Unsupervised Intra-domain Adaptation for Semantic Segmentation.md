# Unsupervised Intra-domain Adaptation for Semantic Segmentation

- `automatically annotated data` has a problem.
    - synthetic data -> real data 
    - directly adapting models from the source data to the unlabeled target data (to reduce the `inter-domain gap`)
    - But result? ==> bad :(
        - there is the large distribution gap among the target data itself(`intra-domain gap`)

## Approach

<img width="300" src="https://user-images.githubusercontent.com/48315997/102793576-7984f980-43ed-11eb-8cbe-ca0d2986d14e.png">

<img width="595" alt="스크린샷 2020-12-22 오전 12 44 17" src="https://user-images.githubusercontent.com/48315997/102794504-d3d28a00-43ee-11eb-9b61-4eb50fde5d1d.png">


two-step self-supervised domain adaptation approach to minimize the inter-domain and intra-domain gap together.

1. inter-domain gap
: separate the target domain into an easy & hard split (using entropy-based ranking function)

2. intra-domain gap
: self-supervised adaption from the easy to hard split
    - segmentation predictions of easy split data (from G_inter) => pseudo labels 로 사용
    - Given easy split data & pseudo labels, hard split data => D_intra는 easy? hard? 판별


## Results

<img width="698" alt="스크린샷 2020-12-22 오전 12 46 37" src="https://user-images.githubusercontent.com/48315997/102794749-26ac4180-43ef-11eb-853c-45c2e9a5cfd4.png">
