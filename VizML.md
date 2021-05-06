# VizML : A Machine Learning Approach to Visualization Recommendation

<img width="600" alt="스크린샷 2021-05-06 오후 11 37 45" src="https://user-images.githubusercontent.com/48315997/117316985-1095ae00-aec4-11eb-9934-cc9110acc841.png">

## Abstract
- ML approach to visualization recommendation
- learns visualization design choices from a large corpus of datasets and associated visualization
    - identify five key design choices (viz. type, encoding type, ...)
    - train models to predict these design choices using 1M dataset-viz. pairs
- NN predicts well



## Problem Formulation
- `representation`
    - are specified using **encodings** that map from data to the retinal properties(position, length, color) of graphical marks(points, lines, rectangles)

<img width="433" alt="스크린샷 2021-05-06 오후 11 41 19" src="https://user-images.githubusercontent.com/48315997/117317548-90237d00-aec4-11eb-8a08-fac0d1997ced.png">

> That is, to create basic visualizations in many grammars or tools, **an analyst specifes higher-level design choices**, which we defne as statements that compactly and uniquely specify a bundle of lower-level encodings. Equivalently, each gram- mar or tool affords a design space of visualizations, which a user constrains by making choices.

Trained with a corpus of datasets ${d}$ and corresponding design choices ${C}$, ML-based recommender systems treat recommendation as an optimization problem,such that predicted $ C_{rec} ∼ C_{max}. $


## Related Work
### Rule-based 
- encode visualization guidelines as collection of "if-then" statements or **rules.**
- sometimes effective, but high cost

### ML-based
- learn the relationship between data and visualizations by training models on analyst interaction
- DeepEye, Data2Vis, Draco-Learn
    - **do not learn to make visualization design choices**
    - trained with annotations on rule-generated visualizations in controlled settings -> limit
- DeepEye
    - combines rule-based visualization generation with models trained to 1) classify Good/Bad 2) rank lists of viz.
    - **learning to rank**
- Data2Vis
    - Seq2Seq Model that maps JSON-encoded datasets to Vega-lite visualization specifications
    > Vega and Vega-Lite are visualization tools implementing a grammar of graphics, similar to ggplot2.
    - 4300 automatically generated Vega-Lite ex.
- Draco-Learn
    - represents 1) visualizations as logical facts 2)design guidelines as hard and soft constraints, SVM
    - recommends visualizations that satisfy these constraints
- `VizML`
    - In terms of **LEARNING TASK**
        - DeepEye learns to classify and rank visualizations
        - Data2Vis learns an **end-to-end** generation model
        - Draco-Learn learns soft constraints weights
        - By learning to **predict design choices**, **VizML models are easier to quantitatively validate**, provide interpretable measures of feature importance, and can be more easily integrated into visualization systems.
    - In terms of **DATA QUANTITY** ...
- BUT 3 ML-Based systems recommend **both data queries and visual encodings**, while VizML only recommends **the latter.**


## Data
### Feature Extracting
[code](https://github.com/mitmedialab/vizml/blob/b36310106791927eaef3831a0cda7abcec598999/feature_extraction/extract.py)

<img width="453" alt="스크린샷 2021-05-06 오후 11 57 01" src="https://user-images.githubusercontent.com/48315997/117319963-c235de80-aec6-11eb-9a3f-f3e8bfff428f.png">

> We map each dataset to 841 features, mapped from 81 single- column features and 30 pairwise-column features using 16 aggregation functions. 

- Each Col. -> 81 single-column features across four categories
- Dimension(D) feature = # of rows in col.
- Types(T) feature = categorical/temporal/quantitative
- Values(V) feature = the statistical & structural properties of the values within a col.
- Names(N) feature = column name


> We distinguish between these feature categories for three reasons. 
- First, these categories let us **organize how we create and interpret features.**
- Second, we can observe the contribution of diferent types of features. 
- Third, some categories of features may be less generalizable than others. 
- We order these categories **(D → T → V → N)** by how biased we expect those features to be towards the Plotly corpus.


> We create 841 dataset-level features by aggregating these single- and pairwise-column features using the 16 ag- gregation functions

### Design Choice Extraction
> Examples of encoding-level design choices include mark type, such as scatter, line, bar; and X or Y column encoding, which specifes which column is represented on which axis; and whether or not an X or Y column is the single column represented along that axis.

> By aggregating these **encoding-level design choices**, we can characterize **visualization-level design choices** of a chart


## Methods
### Feature Preprocessing
### Prediction Tasks
- **Two visualization-level prediction tasks**
    - Dataset-level features to predict visualization-level design
    - 1) Visualization Type[VT]
    - 2) Has Shared Axis [HSA]
- **Three encoding-level prediction tasks**
    - use features about individual columns to predict how theay are visually encoded
    - consider col. indep.
    - 1) Mark Type[MT]
    - 2) Is Shared X-axis or Y-axis [ISA]
    - 3) Is on X-axis or Y-axis [XY]
- For the **VT, MT** tasks, the 2- class task predicts line vs. bar, and the 3-class predicts scatter vs. line vs. bar. 

### Neural Network and Baseline Models
> In terms of features, we constructed four diferent feature sets by incrementally adding the Dimensions (D), Types (T), Values (V), and Names (N) categories of features, in that order. We refer to these feature sets as **D, D+T, D+T+V, and D+T+V+N=All**. The neural network was trained and tested using all four feature sets independently. The four base- line models only used the full feature set (D+T+V+N=All).
