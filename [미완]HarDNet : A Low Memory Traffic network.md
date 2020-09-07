# HarDNet : A Low Memory Traffic network



## Key Idea

- 기존의 metrics들에서의 `inference time` 측정은 부정확하다.
  - 새로운 metric => memory traffic for accessing intermediate feature maps 측정
  - CIO : approximation of DRAM traffic이 될 수 있다.
- **computation, energy efficiency를 위해서는 fewer MACs, less DRAM이 좋은 것임** -> 연구 방향
- 각각의 레이어의 MoC에 soft constraint를 적용했음.
  - low CIO network model with a reasonable increase of MACs를 위해
  - 방법 -> avoid to employ a layer with a very low MoC such as a Conv1x1 layer that has a very large input/output channel ratio.
- Densely Connected Networks에 영감을 받아 모델 빌딩함.
  - DenseNet의 다수의 layer connections들을 줄였음. => concatenation cost를 줄이기 위해
  - **balance the input/output channel ratio by increasing the channel width** of a layer according to its connections.

- DRAM traffic

## Basic Concepts

- MAC : number of multiply-accumulate operations or floating point operations
- DRAM : Dynamic Random-Access Memory
  - read/write model param. and feature maps
- CIO : Convolutional input/output
  - 모든 conv layer에 대해 IN(C,W,H) X OUT(C,W,H) sum
- MoC : MACs over CIO of a layer

## Related Works

- TREND : exploiting shortcuts
- Highway networks, Residual Networks : add shortcuts to sum up a layer with multiple preceeding layers.
- DenseNet : concatenates all precedding layers as a shortcut achieving more efficent deep supervision.
- 그러나 shortcuts는 large memory usage, heavy DRAM traffic을 유발할 수 있다.
  - Using shortcuts elongates the lifetime of a tensor, which may result in frequent data exchanges betwwen DRAM and cache.
- DenseNet의 sparsified version : LogDenseNet, SparseNet
  - Sparse
    - **The pros?** If you have a lot of zeros, you don’t have to compute some multiplications, and you don’t have to store them. So you ***may\*** gain on size and speed, for training and inference (more on this today).
    - **The cons?** Of course, having all these zeros will probably have an impact on network accuracy/performance. But to what extent? You may be surprised.
  - increase the growth rate(output channel widht) to recover the accuracy **dropping from the connection pruning,** and the increase of growth rate **can compromise the CIO reduction**



## Harmonic DenseNet

### Sparsification and weighting

- let layer `k` connect to layer `k-2^n` if 2^n divides k, where n is a non-negative integer and` k-2^n >= 0` 

  - **2^n 개의 layer들이 이런 식으로 processed되면 layer [1 : 2^n -1]는 메모리에서 flush된다.**

  - Power-of-two-th harmonic waves가 만들어짐. 그래서 `Harmonic` 이다. 

  - [그림]

    

- 이 방식은 concatenation cost를 눈에 띄게 감소시킨다.

- layers with an index divided by a larger power of two are more influential than those that divided by a smaller power of two.

  - 많이 connection되니까 당연히 influential 하다.
  - In this model, they amplify these key layers by increasing their channels, which can balance the channel ratio between the input and output of a layer to avoid a low MoC.
    - 이런 key layer들을 **amplify** 했음(channel 수를 늘리면서)
    - layer `l` has an initial growth rate `k`, we let its channel number to be `k * m^n` , where `n` is the max number satisfying that `l` is divided by `2^n`
    - `m`  은 low-dimensional compression factor 역할을 한다. 
    - `m` 을 2보다 작게하면 input channel을 output channel보다 작게 할 수 있다.
      - Empirically, settin `m` between 1.6 and 1.9

2번째 문단이해가 안간다.  그리고 layer 0 은 항상 연결되어야하는거 아닌가?

### Transition and Bottleneck Layers

- `HDB(Harmonic Dense Block)` : the proposed connection pattern forms a group of layers
  - is followed by a Conv1x1 layer as a transition
- HDB의 depth는 2의 제곱수로 설정
  - HDB의 마지막 레이어가 가장 큰 채널수를 가지도록 하기 위해서
- DenseNet -> gradient할 때 모든 레이어를 다 pass함
- HBD with depth L -> pass through at most `log L layers` 
  - degradation을 완화시키기위해, depth-L HDB를 layer L과 all its preceeding **odd numbered layers**  를 concatenation시킨다.
  - 2~L-2의 all even layer들의 아웃풋은 HDB가 한번 끝날때마다 버려진다.

- Bottleneck layer
  - DenseNet에서는 param. efficiency를 위해 매 Conv3x3 layer전에 bottleneck을 두었다.
  - 하지만 HarDnet에서는 위에서 이미 channel ratio(매 레이어마다 input&output 사이의)의 균형을 잡았으므로 bottleneck layer는 쓸모없어진다.
  - 그래서 HBD에서는 Bottleneck layer없이 Conv3x3 for all layers\
- Transition layer
  - inverted trainsition module
    - maps input tensor to an additional max pooling function along with the original average pooling, followed by concatenation and Conv1x1.
    - 50% of CIO를 감소시킴



## Experiments

- CamVid Dataset

  - replace all the blocks in a FC-DenseNet with HDBs
  - the architecture of FC-DenseNet with an encoder-decoder structure and block level shortcuts to create models for sematic segmentation.

  > We propose FC-HarDNet84 as specified in Table 3 for comparing with FC-DenseNet103. **The new network achieves CIO reduction by 41% and GPU inference time reduction by 35%.** A smaller version, FC-HarDNet68, also outperforms FC-DenseNet56 by a 65% less CIO and 52% less GPU inference time.

  

- ImageNet Datasets

- Object Detection



## Discussion

- CIO still failed to predict the actual inference time
  - such as comparing two network models with significantly differnent architectures
- 어쨌거나 DRAM traffic을 강조하고 싶어함.
- traffic reduction을 위한 가장 좋은 방법은 MoC를 증가시키는 것
  - which might be counter-intuitive to the widely-accepted knowledge of that using more Conv1x1 achieves a higher efficiency.

