# (미완)

# PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization

## Overview

> What is the good pre-training objectives tailored for abstractive text summarization?

--> `GSG`

- self- supervised objective
- important sentences are removed/masked from an input doc- ument and are generated together as one output sequence from the remaining sentences, similar to an extractive summary.
    - 중요한 문장은 인풋 과정에서 `masked`되고, 남은 문장들 중에서 extractive summary처럼 가져오는 건가 봄.
- scoring : ROUGE
- `GSG`, `MLM`


## Related Work (Basic Concepts)

- 논문 참조


## Methods

### Architecture

![스크린샷 2020-08-05 오후 12 21 15](https://user-images.githubusercontent.com/48315997/89368717-2b5dab00-d717-11ea-9500-dff52b43c1ad.png)



### Gap Sentences Generation(GSG)

> our proposed pre-training objec- tive involves generating summary-like text from an input document. In order to leverage massive text corpora for pre- training, we design a sequence-to-sequence self-supervised objective in the absence of abstactive summaries.

> we select and mask whole sentences from documents, and concatenate the gap-sentences into a pseudo-summary. The corresponding position of each selected gap sentence is replaced by a mask token [MASK1] to inform the model. Gap sentences ratio, or GSR, refers to the number of selected gap sentences to the total number of sentences in the document, which is similar to mask rate in other works.



- 워드를 마스킹하는 게 아니라 센텐스 단위로 마스킹
    - 이것을 트랜스포머 모델에 넣고 마스킹한 문장을 추론하는 태스크 수행

- 어떻게 하면 중요한 센텐스를 선택해서 마스킹할 수 있는가?
    - ROUGE1-F1 score 사용
    

#### ROUGE

- ROUGE + F1 score
- 정답 센텐스가 있다고 가정하는 것
    - pretraining에 활용하고 싶은 것이기 때문에 정답 센텐스가 없음
    - 셀렉트 센텐스와 나머지 센텐스를 비교해서 ROUGE1-F1 스코어를 구함.
- 기존의 ROUGE 정의와는 약간 다르다.


### Algorithm

![스크린샷 2020-08-05 오후 12 24 22](https://user-images.githubusercontent.com/48315997/89368734-357fa980-d717-11ea-9c8a-2cdc2794e532.png)


### Masked Language Model(MLM)

> Following BERT, we select 15% tokens in the input text, and the selected tokens are (1) 80% of time replaced by a mask token [MASK2], or (2) 10% of time replaced by a random token, or (3) 10% of time unchanged.

- 랜덤하게 특정 단어를 masking 시킴.
    - 인풋을 넣어줄 때 이 단어들을 넣어주지 않는다
- 마스킹된 단어가 무엇인지 추측하는 ...
-> Can get gnereal linguistic knowledge
-> labeling, data를 가질 필요없이 모든 데이터로 사용할 수 있다.

### Pre-training each corpus

- C4
- HugeNews

### Fine-tuning datasets

- XSum
- CNN/DailyMail
- NEWSROOM
- Multi-News
- ...

tfds에서 가져옴.

### Experiments

1. Pre-training ablation experiments to choices of pre-training corpus, objective, and vocabulary size Using PEGASUS_Base instead of PEGASUS_Large

- `Ind-Orig`
- masking : 30%
- Unigram 96K

2. Larger Model Results
- base모델로 찾은 다음에 large모델을 생성

3. Fine-tuning with low-resource
- 적은 데이터셋에 대해서도 성능이 좋다

4. Qualitative Observations
- 사람이 봤을 때 진짜 좋은 성능인지 확인

실험결과 -> 실제로는 GSG 방식만 사용하는 것이 더 좋다.

## Code

## Additional Studies

## References
