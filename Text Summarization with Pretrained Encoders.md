# Text Summarization with Pretrained Encoders

July 26, 2020

https://arxiv.org/abs/1908.08345

---

이 저자가 참여한 전 논문이 `Fine-tune BERT for Extractive Summarization`였음.
- 따라서 발전된 부분에 대해 살펴볼 것임.


## 이전 연구와의 차이점 (Differences from previous studies
)

- 전에는 extractive summarization에 대해서만 살펴봤었음.
- 현 연구에서는 **a general framework for both extractive and abstractive models** (두 서머리 모델을 사용함)



## Data

- CNN/Dailymail : highlighted된... -> extractive summaries
- NYT : abstractive summaries
- XSum : one-sentence summary


## Method

- extractive model
    - built on top of the encoder by stacking several inter-sentence Transformer layers

- abstractive model
    - a new fine-tuning schedule which adopts different optimizers for the encoder and the decoder as a means of alleviating the mismathch between two (the former is pretrained while the latter is not)

- **a two-staged fine-tuning approach**
    - it can further boost the quality of the generated summaries
    - the combination of extractive and abstractive objectives can help generate better summaries (Gehrmann et al., 2018)
    - 즉 두 가지 모델을 컴바인시키면 성능 향상을 노릴 수 있다는 아이디어에서 착안한 연구. 

- architecture of BERTSUM

<img width="858" alt="스크린샷 2020-07-26 오후 5 48 23" src="https://user-images.githubusercontent.com/48315997/88475140-489cb780-cf68-11ea-8a26-bdd9677e1f6a.png">

- Summarization Encoder
    - Interval Segment Embeddings
        - sent(i)를 홀수,짝수 순서에 따라 E(a) or E(b)로 segment embedding 한다.
        ```
        sentence -> [sent1,sent2,sent3,sent4,sent5]
        embedding -> [E(a),E(b),E(a),E(b),E(a)]
        ```

- Extractive Summarization
    - 이전 연구와 수식 동일

- Abstractive Summarization
    - **the encoder is the pretrained BERTSUM and the decoder is a 6-layered Transformer initialized randomly.**
        - encoder & decoder 사이의 mismatch 가능성 있음
            - new fine-tuning schedule!
    - difference optimizer 적용!

- two-stage fine-tuning
    ```
    1. fine-tune the encoder on the extractive summarization
    2. fine-tune it on the abstractive summarization
    ```
    - using extractive objectives can boost the performance of abstractive summarization.


## Results

- ROUGE score 사용함

- 전보다 성능도 많이 좋아짐.

## 관련 논문

<details>
<summary>접기/펼치기 버튼</summary>
<div markdown="1">

- Sebastian Gehrmann, Yuntian Deng, and Alexander Rush. 2018. Bottom-up abstractive summarization. In Proceedings of the 2018 Conference on Empiri- cal Methods in Natural Language Processing, pages 4098–4109, Brussels, Belgium.
- Chin-Yew Lin. 2004. ROUGE: A package for auto- matic evaluation of summaries. In Text Summariza- tion Branches Out, pages 74–81, Barcelona, Spain.
- Shashi Narayan, Shay B. Cohen, and Mirella Lapata. 2018a. Don’t give me the details, just the summary! topic-aware convolutional neural networks for ex- treme summarization. In Proceedings of the 2018 Conference on Empirical Methods in Natural Lan- guage Processing, pages 1797–1807, Brussels, Bel- gium.
- Shashi Narayan, Shay B. Cohen, and Mirella Lapata. 2018b. Ranking sentences for extractive summa- rization with reinforcement learning. In Proceed- ings of the 2018 Conference of the North American Chapter of the Association for Computational Lin- guistics: Human Language Technologies, Volume 1 (Long Papers), pages 1747–1759, New Orleans, Louisiana.
- Alexander M. Rush, Sumit Chopra, and Jason Weston. 2015. A neural attention model for abstractive sen- tence summarization. In Proceedings of the 2015 Conference on Empirical Methods in Natural Lan- guage Processing, pages 379–389, Lisbon, Portugal.
- Xingxing Zhang, Furu Wei, and Ming Zhou. 2019. HI- BERT: Document level pre-training of hierarchical bidirectional transformers for document summariza- tion. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 5059–5069, Florence, Italy. Association for Computational Linguistics.
- Xingxing Zhang, Mirella Lapata, Furu Wei, and Ming Zhou. 2018. Neural latent extractive document sum- marization. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Pro- cessing, pages 779–784, Brussels, Belgium.

</div>
</details>
