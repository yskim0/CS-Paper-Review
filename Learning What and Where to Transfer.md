# Learning What and Where to Transfer

## Overview
### 1. Transfer Learning
> _mitigating the lack of samples_
> * by utilizing the knowledge of pre-trained source models,   
> * improves the performance of a model on a new task   
![transfer](https://i.ytimg.com/vi/uU0hc1z-b4w/maxresdefault.jpg)   

### 2. Meta Learning
> _learning the learning rules to transfer the source knowledge_
> > [ Limitations of previous methods ]
> > * require the same architecture between a source and target models
> > * require exhaustive hand-crafted tuning (ex. attention transfer, Jacobian matching)
   
   : Learning What / Where to Transfer == __L2T-ww__
#### * __Where to Transfer__
A meta-network g decides useful pairs of source/target layers to transfer

#### * __What to Transfer__
A meta-network f decides useful channels to transfer
