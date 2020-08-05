# (미완)

# PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization

## Overview

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

### Algorithm

![스크린샷 2020-08-05 오후 12 24 22](https://user-images.githubusercontent.com/48315997/89368734-357fa980-d717-11ea-9c8a-2cdc2794e532.png)


### Masked Language Model(MLM)

> Following BERT, we select 15% tokens in the input text, and the selected tokens are (1) 80% of time replaced by a mask token [MASK2], or (2) 10% of time replaced by a random token, or (3) 10% of time unchanged.


### Datasets

- XSum
- CNN/DailyMail
- NEWSROOM
- Multi-News
- ...

## Code

## Additional Studies

## References
