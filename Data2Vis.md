# Data2Vis: Automatic Generation of Data Visualizations Using Sequence-to-Sequence Recurrent Neural Networks

## Abstract
- end-to-end trainable neural translation model
- formulate visualization generation as **a language translation problem**, where data specifications are mapped to visualization specifications in a declarative language **(Vega-Lite)**.
    - Vege-Lite -> JSON format
- multilayered ateention-based encoder-decoder network with LSTM
- introduce 2 metrics - language syntax validity, visualization grammar syntax validity


## Related Work
### Declarative Visualization Specification

> One of our aims with Data2Vis is to bridge this gap between the speed and expressivity in specifying visualizations.

### Automated Visulaization

> We pose visualization specifica- tion as a machine translation problem and intro- duce Data2Vis, a deep neural translation model trained to automatically translate data specifica- tions to visualization specifications. Data2Vis emphasizes the creation of visualizations using rules learned from examples, without resorting to a predefined enumeration or extraction of con- straints, rules, heuristics, and features.
- **Machine Translation Problem**

### DNNs for Machine Translation

> Data2Vis is also a sequence- to-sequence model using the textual source and target specifications directly for translation, with- out relying on explicit syntax representations.


## Model

<img width="754" alt="스크린샷 2021-05-07 오전 1 43 49" src="https://user-images.githubusercontent.com/48315997/117335052-adad1280-aed5-11eb-9188-0b30c1cb9533.png">

- the data visualization problem as a **Seq2Seq translation problem**
```
input : dataset (fields, values in JSON format)
output : valid Vega-Lite visualization specification
```
- **encoder-decoder archi.**
> where the encoder reads and encodes a source sequence into a fixed length vector, and a decoder outputs a translation based on this vec- tor.

- **Attention** Mechanism
> Atten- tion mechanisms allow a model to focus on aspects of an input sequence while generating out- put tokens.

- **Beam Search algorithm**
> The beam search algorithm used in sequence-to-sequence neural translation models keeps track of k most probable output tokens at each step of decoding, where k is known as the beamwidth. This enables the generation of k most likely output sequences for a given input sequence.

- THREE techniques : **bidirectional encoding, differential weighing of context via an attention mechanism, and beam search**

- **character**-based sequence model

## Data and Preprocessing

1. the model must select a subset of fields to focus on when creating visual- izations (most datasets have multiple fields that cannot all be simultaneously visualized)
2. the model must learn differences in data types across the data fields (numeric, string, temporal, ordinal, categorical, etc.), which in turn guides how each field is specified in the generation of a visualiza- tion specification.
3. the model must learn the appropriate transformations to apply to a field given its data type (e.g., aggregate transform does not apply to string fields).

- view-level transforms : aggregate, bin, calculate, filter, timeUnit
- field-level transforms : aggregate, bin, sort, timeUnit


## Evaluation Metrics
- language syntax validity(lsv)
    - measure of how well a model learns the syntax of the underlying language used to specify the visualization.
- grammar syntax validity(gsv)
    - a measure of how well a model learns the syntax of the grammar for visualization specification.


## Experiments

### Results
<img width="887" alt="스크린샷 2021-05-07 오전 1 49 00" src="https://user-images.githubusercontent.com/48315997/117335706-66735180-aed6-11eb-90cc-95b38ea1224a.png">

<img width="765" alt="스크린샷 2021-05-07 오전 1 49 10" src="https://user-images.githubusercontent.com/48315997/117335723-6bd09c00-aed6-11eb-8c0e-9e9d0e1348a2.png">


## Limitations

- Field Selection and Transformation
- Training Data


## Future Work
- Training Data and Training Strategy
- Extending Data2Vis to Generate Multiple Plausible Visualizations
- Targeting Additional Grammars
- Natural Language and Visualization Specification
