# Fine-tune BERT for Extractive Summarization

July 25, 2020

https://arxiv.org/pdf/1903.10318.pdf

---

## Summarization의 종류

- abstractive summarization : contains words or phrases that **were not in the original text...**
    - paraphrasing와 유사

- extractive summarization : by copying and concatenating the most important spans in a document
    - 중요한 문장을 copy & paste 하는 것으로 서머리를 만듦.

- 이 논문에서는 후자인 extractive summarization을 사용

## Data

- CNN/Dailymail 
- NYT

## Method

### Encoding Multiple Sentences

![IMG_55CE1AC3669E-1](https://user-images.githubusercontent.com/48315997/88408018-ef207580-ce0d-11ea-846f-ea3d451f8cc3.jpeg)


- **insert a [CLS] token before each sentence and a [SEP] token after each sentence**
    - *In vanilla BERT, [CLS] is used as a symbol to aggregate features from one sentence or a pair of sentences.*
    

### Interval Segment Embeddings

- sent(i)를 홀수,짝수 순서에 따라 E(a) or E(b)로 segment embedding 한다.


```
sentence -> [sent1,sent2,sent3,sent4,sent5]
embedding -> [E(a),E(b),E(a),E(b),E(a)]
```

### Fine-tuning with Summarization Layers

- simple classifier : sigmoid function
- inter-sentence transformer
    - extracting document-level features focusing on summarization tasks from the BERT outputs
    - 공식에 layer normalization과 multi-head attention operation를 이용한다 하는데 MHAtt는 더 알아봐야할 듯함.
- Recurrent Neural Network
    - RNN이 더 좋게 만들 수 있음(왜지)
    - BERT output에 LSTM 레이러를 적용시켜서 summarization-specific features를 학습할 수 있도록 함.

## Results

- ROUGE score 사용함

<img width="374" alt="스크린샷 2020-07-25 오전 12 37 36" src="https://user-images.githubusercontent.com/48315997/88408821-0b70e200-ce0f-11ea-9103-8a495d7475df.png">

