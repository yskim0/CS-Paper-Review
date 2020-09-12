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

* * *

### Meta-learning based Transfer-learning: Learning What / Where to Transfer(L2T-ww)
1. selective transfer depending on a source and target task relation
2. effective training scheme that learns meta-networks and target model simultaneously
3. applicable between heterogeneous and multiple network architectures and tasks


## Related Works
< Transfer Learning Schemes >
1. Learning without Forgetting (LwF)
2. Attention Transfer (AT)
3. Unweighted Feature Matching (FM)


## Methods
<img width="394" alt="method" src="https://user-images.githubusercontent.com/49134038/92984898-c69d2980-f4e8-11ea-9227-94b9566bb831.png">

### 1. What to Transfer
__== Channel Importance__
A meta-network f decides useful channels to transfer   
![what](https://user-images.githubusercontent.com/49134038/92984782-cea89980-f4e7-11ea-98ca-88cc5ea2e27e.png)   
=> choose important channels for learning a target task

### 2. Where to Transfer
__== Pair Importance__
A meta-network g decides useful pairs of source/target layers to transfer   
![where to](https://user-images.githubusercontent.com/49134038/92984788-d49e7a80-f4e7-11ea-8dfc-160a16e6edfa.png)   
=> choose pairs of feature-matched layers among all the possible pairs

### Training Meta-Networks
> Total Loss
> <img width="301" alt="loss" src="https://user-images.githubusercontent.com/49134038/92985056-1cbe9c80-f4ea-11ea-9805-0974cf475821.png">   

_Original Bilevel Scheme [4,5] for training meta-parameters φ_   
<img width="314" alt="ori" src="https://user-images.githubusercontent.com/49134038/92984920-0a902e80-f4e9-11ea-9ede-ece70e901c33.png">   
- transfer loss (L_wfm) acts as a regularization
- a large number of steps T is required to obtain meaningful gradients -> time-consuming   

_Proposed Bilevel Scheme for training meta-parameters φ_    
<img width="290" alt="updated" src="https://user-images.githubusercontent.com/49134038/92985099-7aeb7f80-f4ea-11ea-962a-97cac52310a7.png">   
- effective for learning φ with a small number of steps T -> reduced time
- learns θ and φ simultaneously without separate meta-learning phase   

* * *

#### Learned Matching
<img width="182" alt="model" src="https://user-images.githubusercontent.com/49134038/92985194-593ec800-f4eb-11ea-92c4-9c7696beff82.png">


## Experiments
1. with Various Tasks and Architectures
<img width="709" alt="result1" src="https://user-images.githubusercontent.com/49134038/92985278-16312480-f4ec-11ea-91f7-514da0e65824.png">   

* Learning __WHAT to Transfer__ improves all the baselines
* Learning __WHERE to Transfer__ gives more improvements on what to transfer

2. with Different Architectures, Initializations, and Datasets
<img width="602" alt="result2" src="https://user-images.githubusercontent.com/49134038/92985280-1a5d4200-f4ec-11ea-86f3-16d85621ea39.png">   

3. with Limited Data-Regime(Settings) Experiments
<img width="219" alt="result2 2" src="https://user-images.githubusercontent.com/49134038/92985281-1b8e6f00-f4ec-11ea-9be5-3ee34c3f522d.png">   

* Smaller the volume of the target dataset, more relative gain of the results
* In other words, efficiency boosts up the performance of a target model

4. Saliency Map comparing unweighted(FM) and weighted(L2T-w)
<img width="451" alt="result3" src="https://user-images.githubusercontent.com/49134038/92985282-1cbf9c00-f4ec-11ea-80c7-742d463f48c2.png">   

* more and less activated pixels in L2T-w
