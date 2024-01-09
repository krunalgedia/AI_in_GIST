# AI_in_GIST

## Overview

Brief overview of the project.

## Table of Contents
- [Preprocessing](#Preprocessing)
  - [Variable Encoding](#Variable-Encoding)
  - [Handle Missing Values](#Handle-Missing-Values)
  - [Detect Outliers](#Detect-Outliers)
  - [Handle Outliers](#Handle-Outliers)
  - [Dimension Reduction Techniques](#Dimension-Reduction-Techniques)
  - [Handle Skew data](#Handle-Skew-data)
  - [Techniques to handle imbalance dataset](Techniques-to-handle-imbalance-dataset)
  - [Types of Bias](#Types-of-Bias)
  - [Goodness of fit test](#Goodness-of-fit-test)
- [Classical ML](#Classical-ML)
  - [Regression](#Regression)
  - [Classification](#Classification)
  - [Time Series](#Time-series)
  - [Clustering](#Clustering) 
- [DL Papers](#DL-Papers)
  - [Fully Convolution Neural Network](#Fully-Convolution-Neural-Network)
  - [Two Stage Object detectors](#Two-Stage-Object-detectors)
  - [Vision Transformers](#Vision-Transformers)
  - [NLP/Language Models](#NLP/Language-Models)
  - [Document AI](#Document-AI)


## Preprocessing

### Variable Encoding

| Encoding Type      | Categorical Ordinal Var     | Categorical Nominal Var   | Description                                      |
|--------------------|-------------------|-------------------|--------------------------------------------------|
| Classical          | One Hot      | One Hot                 | 1 when True otherwise 0, use when n(unique) is small/reasonable |
|    | Label/ordinal  | -    | Denotes hierarchical levels |
|     | Hashing                 | Hashing   | Hash function mapping for each unique cats, can handle large cardinality/n(unique) |
|     | Binary                | -    | One hot + Hashing |
|  |  | Frequency | count number of unique |
|  Bayesian| Target |Target | Encode target info in encoding using conditional target value for each unique cat value. Can lead to overfitting so smoothed versions exist.|
|  | LOO Target | LOO Target| Exclude the current row while calculating target encoding for that row. |


|Numerical Encoding| Description|
|---|-----|
|Binning equal|Bin equally, replace the value with bin number. Depends on the numerical variable, problem at hand|
|Binning uneuqal|Bin unequally, replace value with bin number. Depends on the numerical variable, problem at hand|
|Binning quantile|Bin in quantiles of values of the variable, replace the value with bin number. Depends on the numerical variable, problem at hand|

### Handle Missing Values
| Classification   | Regression        | Time Series       | Clustering        |
|------------------|-------------------|-------------------|-------------------|
| Deletion                |   Deletion                |  Deletion                 | Deletion                  |
| Conditional Imputation with mode        | Conditional Imputation with mean/median     | Forward/Backward fill         | Kmeans Conditional imputation         |
| -        |  Interpolation (linear, spline, etc)         | Interpolation (linear/spline)         | -        |
| Treat as special category like UNK        |     -      |   -       |     Treat as saparate cluster    |
|  -       |     -      |    Change bining to reduce effect (monthly-> quaterly)      |  -       |

### Detect Outliers
| Method             | Description        |
|--------------------|--------------------|
|  Inter-quartile range (IQR)                  |  Statistical method to detect outliers using quantiles                  |
|  Isolation forest                  |  Random forest, easy to detect outliers since they need fewer cuts/branches and can be separated easily.                  |
|  One Class SVM                  | Separate outlier cluster from inline cluster since another class is the outlier, use kernel like rbf                   |
|  DBSACN                  |  Clustering Algo, works for anomaly detection                  |
|  Auto-encoder                  |  Deep Learning method to detect vectors away from normal embedding vector, thus larger loss for them                  |

### Handle Outliers

| Method             | Description        |
|--------------------|--------------------|
| Deletion                 | Do only if you know the reason                   |
| Transformation                   |  Can lead to loss of info present in the shape of input variable                  |
| Truncate                   | Replace by the threshold                   |
| Binning                   | Binning numerical vars                   |
| Robust model                   | Like ensemble methods                   |
| Robust loss                   | Like Huber loss                   |

### Dimension Reduction Techniques

| Method             | Description        |
|--------------------|--------------------|
| Feature Selection  | Filter (remove correlation features, not imp features based on feature importance)                   |
|             | Subsets (RFE & forward selection)               |
|             | Train loop (LASSO)               |
| Feature Extraction | PCA (linear), kernal PCA (like rbf kernel), t-SNE (nonlinear), UMAP (better for nonlinear, large dataset)               |
|             | LDA (supervised ML, max separability of class directions)               |
|                    |  Auto encoder (DL) as in clustering                  |
| Clustering                    | Making groups for cats and numerical                   |
| Binning                   | Binning for numerical and cats                  |
| Aggegration features and transform                   | ex: Polynomial features, interaction terms                   |

### Handle Skew data

| Function Transformation | Power Transformation | Quantile Transformation |
|--------------------------|----------------------|--------------------------|
| Logarithmic (>0 Right skewed input)             | Box-Cox (Input should be >0)             | Quantile Normalization  |
| Squared (Left skewed input)             | Yeo-Johnson          | Rank Transformation      |
| Square Root (>0 right skewed input)            |                    |                         |
| Reciprocal (>0 strong right skewed input)            |                    |                         |

### Techniques to handle imbalance dataset
| Method             | Description        |
|--------------------|--------------------|
| Oversample, undersample | oversample minority class with replacement, undersample/remove entries from majority class  |
| BalancedML | Like BalancedBaggingClassifier, does oversampling |
| SMOTE | Synthetically generate new samples using KNN |
| Appropriate ML | like tree ensemble methods |
| Better metric | Precision, Recall, F-beta score instead of accuracy |
| CV | Stratified cross-validation scores, bootstrapping, etc |

### Types of Bias
|Bias|
|---|
| Inherent bias in data|
| Sampling/data collection bias|
| Preprocessing bias|
| ML algo bias, like no L1/L2 regularizer, etc|
| ML algo evaluation metric bias|

### Goodness of fit test
| Test             | Description        |
|--------------------|--------------------|
| Chi-squared | Between two categorical variables like compare unbiased or biased coin toss distribution |
| Kolmogrow Smirnov | Continuous variables, >2000 data, a non-parametric test (no underlying hypothesis) |
| Anderson Darling | Similar to KS, gives more importance to tails |
| Shapiro Wilk | checks if data is gauss (compares quantile to that of gauss), <2000 data  |
| AIC | Regression tasks, continuous variables, Combines GOF+model complexity (DOF)  |
| BIC | Like AIC but Bayesian in approach, takes the number of prior data into account, more penalty for model complexity |
| R squared | Continuous variables, MSE compared with Mean/Intercept only model MSE |
| Adjusted R squared | Similar to R squared but takes model complexity into account (DOF) |


## Classical ML

### Regression & classification

| Method             | Description        |
|--------------------|--------------------|
| Linear Regression/OLS | When x~y linearly, residual is homoscedastic, and variance(residual)=constant |
| Linear Regression extension | Adding non-linear and interaction terms |
| Generalised linear models | For GLM family, check y<br> Count data-> Poisson<br> -ve Binomial<br> Continuous->Normal<br> Conti. right skew: Gamma<br> Conti. left skew: Inverse Gauss<br> Probability distribution: Binomial, Multinomial  |
||For GLM link function, check y ~ x<br> y~x -> Identity<br> ln(y)~x->Log link<br> logit(y)~x: Logit link|
| Decision trees | Classification:  |
| Random forest |  |
| XGBoost |  |
| Adaboost |  |
| SVM |  |
| Kernal SVM |  |
| LDA |  |

### Time Series

### Clustering





## DL Papers

### Fully Convolution Neural Network

| Paper                                | Date       | Description                                      |
|--------------------------------------|------------|--------------------------------------------------|
| [LetNet-5](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=726791)        | 1988 | CNN-AvgPool NN for softmax MNIST classifier, 1st usage of CNNs & weight sharing idea, activations: tanh, softmax  | 
| [AlexNet](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)                     | 2012 | 1000 class classifier, Similar to LeNet but deeper, 1st usage of Dropout, ReLU    | 
| [VGGNet-16/19](https://arxiv.org/abs/1409.1556) | 2014 | Very deep (16/19 layers) and narrow CNN, large number of small filters 3X3 to capture more complex and granular features, stacking of many layers  | 
| [InceptionNet](https://arxiv.org/abs/1409.4842) | 2014 | Stacking of parallel modules, 1X1 followed by either 3X3 or 5X5, or 1X1 modules. 1X1 reduces channels dim & time (bottleneck layers) and diff nXn extracts low & high-level features. Global avg pooling counters overfitting & no. of params, Vanishing gradient countered by 2 auxiliary classifiers using same labels (0.3 weighted in final loss).                                              |
| [InceptionNetv2 & v3](https://arxiv.org/abs/1512.00567) |     | Factorize nXn conv to 1Xn and nX1, thus improve speed. BN in auxiliary classifier, label smoothing for overfitting, RMSProp optimizer       |
| [ResNet](https://arxiv.org/abs/1512.03385)    |                              | Residual connection in CNN blocks of deep CNN for vanishing grads.                                                               |
| [InceptionNetV4 and InceptionResNet](https://arxiv.org/abs/1602.07261) |        | More uniform inception modules with new dim reduction blocks, skip connect module o/p to i/p (same channel dim achieved by 1X1) & replaced pooling operation. Scale residual activation to 0.3 to decrease vanishing grads.                              |
| [DenseNet](https://arxiv.org/abs/1608.06993)   |     | Each layer in block receives input from all previous layers counter vanishing grads, transition layers between dense blocks reduce spatial dim to balance computational cost and accuracy.                                                             |
| [Xception](https://arxiv.org/abs/1610.02357) |   2017      | Based on Inception v3, instead of inception modules,  it uses depthwise separable (depth + pointwise) convolutions over input image (grouped convolutions: i/p channel-wise conv, concatenate and apply pointwise 1X1 conv)                                                             |
| [ResNeXt](https://arxiv.org/abs/1608.06993)      | 2016        | CNN with several Depthwise separable convolutions with ResNet.                                                            |
| [MobileNetv1](https://arxiv.org/abs/1704.04861)      | 2017       | Depthwise separable convolutions network for mobile and embedded devices.                                                        |
| [MobileNetv2](https://arxiv.org/abs/1801.04381)      | 2018    | [Inverted Residual](https://towardsdatascience.com/mobilenetv2-inverted-residuals-and-linear-bottlenecks-8a4362f4ffd5) connections (skip in narrow-wide-narrow type block with wide obtained by depthwise separable convolutions). Loss of extra info due to ReLU not coupled by wide-narrow-wide channel type block, linear activation in used in the final layer of block (linear bottleneck)                   |
| [MobileNetv3](https://arxiv.org/abs/1905.02244)  |   | AutoML tools, MnasNet to select coarse architecture using reinforcement learning and NetAdapt to fine tune. Use [squeeze-and-excitation](https://medium.com/@tahasamavati/squeeze-and-excitation-explained-387b5981f249) block (per channel weights using global avg pooling and NN, apply to original channel), remove 3 expensive layers from v2.                                                              |
| [EfficientNet](https://arxiv.org/abs/1905.11946)  |2020      | AutoML automated neural architecture search (NAS) to select balanced dim of width, depth and resolution (compound scaling method). Usage of efficient building blocks like inverted residuals, linear bottleneck                   |

### Two Stage Object detectors
| Paper                                | Date       | Description                                      |
|--------------------------------------|------------|--------------------------------------------------|
| [RCNN](https://arxiv.org/abs/1311.2524) |  | Selective search for regions proposal (~2000), warp each RoI and pass each through CNN for feature extraction, linear SVM for classification and offset bbox regression. Non-max suppression to select from >1 bbox of same object                |
| Fast RCNN   | ...        | Pass entire image through large backbone CNN, selective search on output, crop, wrap each RoI and do RoI pooling and pass through small detection head NN for offset bbox regression and classification and objectness score. Faster inference due to usage of truncated SVD on model weights to retain imp weights/nodes from head. About 25X faster than RCNN.                            |
| Faster RCNN   | ...        | Replace selective search in Fast RCNN by Region proposal neural network (RPN), 9 anchors different shape boxes to select best bbox per object.  About X10 faster than Fast RCNN       |
| [Mask RCNN](https://arxiv.org/abs/1703.06870)     | ...        | Add extra output to Faster RCNN to perform instance segmentation to classify each RoI.|
| [Cascade RCNN](https://arxiv.org/abs/1712.00726) | ...        | Detection head bbox output depends on RPN performance in Faster RCNN, coupled by poor inference conditional on IoU. To counter, cascade of individual RPN is used each trained using the preceding bbox RPN prediction trained for increasing IoU. |        

### Vision Transformers
| Paper                                | Date       | Description                                      |
|--------------------------------------|------------|--------------------------------------------------|
| [Vision Transformer](https://arxiv.org/abs/2010.11929)    | October 2020       | Encoder block of original transformer. Segment images into patches, flatten each to pass and use them as tokens for trainable embedding layer (patch embeddings),  and treat them as tokers. Standard learnable 1D position embeddings (no gains with 2D-aware position embeddings). MLP layers with GELU non-linearity. CLS token for image classification. SOTA results when trained on large data (14M-300M images)                                                        |


### NLP/Language Models

| Paper                                | Date       | Description                                      |
|--------------------------------------|------------|--------------------------------------------------|
| [Transformer](https://arxiv.org/abs/1706.03762)    | 	June 2017   | Encoder-decoder model, Multi-head self-attention (Query, Key, Value) in encoder and Masked multihead attention MLM and multi-head cross attention (K,V from encoder output, V from decoder), skip connections, [Layer normalisation](https://stats.stackexchange.com/questions/474440/why-do-transformers-use-layer-norm-instead-of-batch-norm) (across all features of a token), BPE used as tokenizer, trainable text 512 length embedding, fixed non-trainable absolute position encoding (sin/cos), trained on WMT English-German, English-French dataset.    |
| [BERT](https://arxiv.org/abs/1810.04805)    | October 2018 | Encoder transformer model, max input 512 tokens, WordPiece tokenizer, trainable text embedding length 768, fixed non-trainable absolute position encoding (sin/cos), multi-head self-attention (Query, Key, Value), final linear-relu-softmax layer, trained on left-right context (bidirectional embedding), Pre-training (MLM task (MASK token), Next Sentence Prediction NSP task using CLS-sent1-SEP-sent2-SEP tokens), BERT fine-tuning (Text Classification CLS token, Question-Answer Task CLS-ques-SEP-context-SEP).   |
| [RoBERTa](https://arxiv.org/abs/1907.11692)    | July 2019        | Optimize BERT, Dynamic masking of tokens in MLM in training pre-training BERT, no NSP due to low value, ByteEncoder, X10 train data (160GB), larger batch size    |
| [ELECTRA](https://arxiv.org/abs/2003.10555)    |  March 2020       | Replaced Token Detection instead of MLM (Generator BERT predicts MSK tokens, Discriminator BERT predicts isOriginal or isReplaced) thus focus on all tokens in sequence and not just MSK token in BERT, no NSP        |
| [ALBERT](https://arxiv.org/abs/1909.11942)    | 	September 2019 | Cross param sharing across BERT blocks, Embedding matrix factorization (AXB=(AXN)*(NXB)) leading to 1/10 size of BERT. Pre-training MLM and Sentence order prediction SOP, no NSP since it makes more sense to know inter-sentence coherence.        |                                      
| [DistilBERT](https://arxiv.org/abs/1910.01108)   | October 2019        | 60X faster, 0.40 size, 97% BERT accurate. Teacher-student knowledge distillation. Train: train teacher BERT for MLM task. Student BERT soft prediction with teacher BERT soft target using KL divergence loss, student BERT hard prediction with hard target using CE loss and cosine similarity loss between soft target and soft prediction embeddings.|
| [TinyBERT](https://arxiv.org/abs/1909.10351)    | 	September 2019     | Reduce Student BERT to 4 encoder block, 312 emb size (Teacher BERT has 12 blocks, 786 emb size). Knowledge distillation is not only at Prediction layer (as in DistilBERT), but also Embedding layer and transformer layer by minimizing MSE between them. Dimension mis-match b/w S and T solved by matrix factorization S(NX312) x Weight(312 X 768) = T(Nx768). Learn weight matrix also.                                             | 
| [GPT](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)    | June 2018       | Generative pre-training and discriminatory fine tuning. Architecture is similar to the decoder of original transformer. Unsupervised pre-training (thus able to use large corpus) using long Book texts (thus capture long range info) for standard language model objective of maximizing likelihood of sum(log P(xi|x1..x_i-1)). The pre-trained language model supervised fine tuned for text classification, text entailment x=(premise$hypothesis)->y=class, similarity x=(sent1$sent2) and x=(sent2$sent1)->y=1/0, question and answer mcq x(context$ans1), x(context$ans2)->softmax(2).   |
| [GPT-2](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)  | February 2019  | Demonstrated 0-shot learning (i.e. it learns language processing task without explicit supervision) with Large LM possible. Used new dataset WebText, modified BPE tokenizer, minor modification to decoder, context size 512->1024, 1.5B params. SOTA perplexity on language modelling (predict x_t|x1..x_t-1), accuracy on NER, perplexity/accuracy on LAMBDA data (predict last word of long range sent), accuracy on common sense reasoning, not SOTA ROUGE F1 in summarisation, BLUE in translation, comparable in accuracy in question-answer. These evaluation done without explicit training (fine tuning)|
| [GPT-3](https://arxiv.org/abs/2005.14165)  | May 2020  |  175B params, Auto-regressive language model. Similar model like GPT-2. SOTA on Language model, 35->20 on perplexity, accuracy score on 0 shot, one-shot, few shot; in question answer in 1/3 dataset over T5, not best for translation since 93% training in English, mixed for common sense reasoning, arithmetic task performance improves with shots |
|   |   |    |


### Document AI

| Paper                                | Date       | Description                                    |            
|--------------------------------------|------------|--------------------------------------------------|
| [LayoutLM](https://arxiv.org/abs/1912.13318)    | December 2019        | Uses BERT as backbone model, takes trainable text embedding and new additional 4 positional embeddings (bbox of each token) to get layoutLM embedding. Text and bbox were extracted using pre-built OCR reader. At the same time, the ROI/bbox text in image form is passed through Faster-RCNN to get image embeddings. They are then used with LayoutLM emb. for fine-tuning tasks. Pre-trained for Masked Visual Language Model MLVM to learn text MSK using its position amb and other text+pos emb, Multi-label document classification to learn document representation in CLS token. Fine-tuned for form understanding task, receipt understanding task, and document image classification task i.e. text, layout info in pre-training and visual info in fine-tuning. WordPiece tokenizer, BIESO tags for token labeling.                      |
| [LayoutLMv2](https://arxiv.org/abs/2012.14740)    | December 2020        | Multi-modal emb with Spatial-Aware self-attention Mechanism (self-attention weights with bias vector to get relative spatial info). Pretraining using text, layout and image emb. Text emb = token emb + 1D position emb for token index + segment emb for different text segments. Visual emb = flatten features from ResNeXt-FPN + 1D position emb for token index + segment emb for different text segment, Layout emb - 6 bbox co-od. Pre-training: MVLM, Text image alignment TIA: cover images of token lines, predict covered or not, Text-Image matching: CLS token to predict if image is from the same text. Fine tune: Document image classification, token-level classification, visual question answering on document images.                                                                   |
| [LayoutLMv3]()    | April 2022        | ...                                                                   |
