# AI_in_GIST

## Table of Contents
- [Preprocessing](#Preprocessing)
  - [Handle Missing Values](#Handle-Missing-Values)
  - [Detect Outliers](#Detect-Outliers)
  - [Handle Outliers](#Handle-Outliers)
  - [Dimension Reduction Techniques](#Dimension-Reduction-Techniques)
  - [Handle Skew data](#Handle-Skew-data)
  - [Techniques to handle imbalance dataset](Techniques-to-handle-imbalance-dataset)
  - [Variable Encoding](#Variable-Encoding)
- [Statistics](#Statistics)
  - [Distributions](#Distributions)
  - [Correlation](#Correlation)
  - [Hypothesis Tests](#Hypothesis-Tests)
  - [Types of Bias](#Types-of-Bias)
  - [Goodness of fit test](#Goodness-of-fit-test)
- [Classical ML](#Classical-ML)
  - [Regression & classification](#Regression-&-classification)
  - [Time Series](#Time-series)
  - [Clustering](#Clustering) 
- [DL Papers](#DL-Papers)
  - [Fully Convolution Neural Network](#Fully-Convolution-Neural-Network)
  - [Two Stage Object detectors](#Two-Stage-Object-detectors)
  - [Vision Transformers](#Vision-Transformers)
  - [NLP/Language Models](#NLP/Language-Models)
  - [Document AI](#Document-AI)


## Preprocessing



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
|  Isolation forest                  |  Unsupervised, Random forest, easy to detect outliers since they need fewer cuts/branches and can be separated easily. Need prior knowledge of %outliers. |
|  One Class SVM                  | Unsupervised, One cluster is the normal class, other comprises only the origin point. Maximize the margin. Better to use RBF kernel |
|  DBSACN                  |  Clustering Algo, works to detect anomoly cluster (cluster is least points)  |
|  Auto-encoder                  |  Deep Learning method to detect vectors away from data embedding vector in bottleneck layer, thus a larger loss for them                  |

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
|             | Subsets (RFE (use DT and remove highest gini feature one by one till desired features reached)& Wrapper techniques like forward, backward, bidirectional (same as RFE but you choose eliminator model))               |
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

### Variable Encoding

| Encoding Type      | Categorical Ordinal Var     | Categorical Nominal Var   | Description                                      |
|--------------------|-------------------|-------------------|--------------------------------------------------|
| Classical          | One Hot      | One Hot                 | 1 when True otherwise 0, use when n(unique) is small/reasonable |
|    | Label/ordinal  | -    | Denotes hierarchical levels |
|     | Hashing                 | Hashing   | Hash function mapping for each unique category, can handle large cardinality/n(unique) |
|     | Binary                | -    | One hot + Hashing |
|  |  | Frequency | count number of unique |
|  Bayesian| Target |Target | Encode target info in encoding using conditional target value for each unique cat value. Can lead to overfitting so smoothed versions exist.|
|  | LOO Target | LOO Target| Exclude the current row while calculating target encoding for that row. |


|Numerical Encoding| Description|
|---|-----|
|Binning equal|Bin equally, replace the value with bin number. Depends on the numerical variable, problem at hand|
|Binning uneuqal|Bin unequally, replace value with bin number. Depends on the numerical variable, problem at hand|
|Binning quantile|Bin in quantiles of values of the variable, replace the value with bin number. Depends on the numerical variable, problem at hand|

## Statistics

### Distributions

| Distribution | Question it addresses | Parameters|
|--------------|--------------|------|
|Gaussian| Param mean, standard deviation | | 
|Bernoulli| Probability of success (p) in a single trial| Mean=p, var=p(1-p) |
|Binomial| Probability of k successes in n trials if p is the probability of success in 1 trial. Models number of successes in fixed number of trials.|Mean=np, var=np(1-p) |
| Negative Binomial | Probability of k trials required for fixed r successes, if p is the probability of success in 1 trial. Models number of trials required for fixed no. of successes. | Mean=r/p, var=r(1-p)/p^2 |
|Poisson|Probability of **x events** to occur **in unit** time interval if **lambda events** occur on average in unit time interval. Here, events (discrete var is x axis)| Mean=lambda, var=lambda|
| Geometric | Probability of **an event**/success occurs **after x** number of trials (failures, discrete var), if one event/success occurs on average after **1/p** trials. Here p is the Bernoulli probability of success| Mean=1/p, var=(1-p)/p^2 |
|Exponential| Probability of **an event** to occur **after x** time interval (or any continuous variable like price/distance) if one event occurs on average after **1/lambda** time interval. Here time (continuous var is x axis). Continuous equivalent of geometric distribution| Mean=1/lambda, var=1/lambda^2|
| Gamma | Probability of **alpha events** to occur **in x** time interval if one event occurs in **1/beta** time interval. Here time (continuous var) is x axis. Generalised case of Exponential distribution. | Mean=alpha/beta, var=alpha/beta^2 |

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

### Correlation
| Var1 | Var2  | Test  |
|---|---|---|
| Categorical | Categorical | Chi-square test |
| Categorical | Numerical | t-test, z-test, ANNOVA |
| Numerical | Numerical | Pearson corr (linear), Spearman corr (rank, monotonic) |

### Hypothesis Tests

| Test | Description|
|------|------------|
| Shapiro-Wilk  |   |
|   |   |
|   |   |
| t-test  |   |
| z-test |   |
| ANNOVA |   |
|   |   |
|   |   |
|   |   |
|   |   |
|   |   |
|   |   |
|   |   |
|   |   |


### Hypothesis Tests


## Classical ML

### Regression & classification

| Method             | Description        |
|--------------------|--------------------|
| Linear Regression/OLS | When x~y linearly, residual is homoscedastic, and variance(residual)=constant |
| Linear Regression extension | Adding non-linear and interaction terms |
| Generalised linear models | For GLM family, check y <br>  Count data-> Poisson, Negative Binomial<br>  Continuous->Normal<br>  Continuous right skew: Gamma<br>  Continuous left skew: Inverse Gauss<br>  Probability distribution: Binomial, Multinomial  |
|  |For GLM link function, check y ~ x <br>  y~x -> Identity<br>  ln(y)~x->Log link<br>  logit(y)~x: Logit link|
| Naive Bayes| Mainly for classification problems, calculates Bayes probability for each of the target classes for the given test example and checks which target has the highest probability. If the input variable is Gaussian distributed, can use Gaussian NB, similarly Multi-nomial NB.|
|Logistic Regression| Classification algorithm where you do linear regression with target as logit(p) where p is the probability of being class1 in binary classification|
| Decision trees | Variable, split, SSE or gini index wrt target <br> Classification: Split variable having minimum Gini, max information gain, or minimum entropy <br> Regression: Split variable at a boundary giving minimum SSR in its final leaf nodes|
| Random forest | Bagging + Feature Selection i.e. training several DT each with randomly selected (default) sqrt(features) for classification and f/3 for regression. |
| XGBoost | Similar to random forest. Except for minimum gini score or SSE of a variable and splitting point, we use the similarity score of the variable and splitting point. Similarity score takes into account not only residual but also the number of data in each leaf used to calculate SSR of that leaf (in DT, number of points in leaf is not taken into account). Then it defines info gain as sum(SS_child)-SS(parent) to decide splitting if gain>gmamma, where gamma is the regularisation factor. Further, even the SS calculation has a regularisation param lambda. i.e DT->XGboost=>SSR/Gini->SS=SSR+lambda+no of data in leaf. <br> Also it is fast because it searches quantile for splitting (like qLoRA), parallel cuts eval, caching|
| Adaboost | Make stump (DT with depth=1) for each var and choose var with the least gini. Give it a weight (%correct). Now, remake the dataset with giving more weight(or copy data) which got incorrectly classified and redo the making of stump and weighting. |
| SVM | Generally used in classification Maximize margin such that examples are classified correctly (classification) or target y values deviates less than epsilon from regression line/curve <br> Hard SVM: Maximize margin such that classes are on either side of the decision boundary. <br> Soft SVM: Maximize margin and weighted penalize incorrect classification from its decision from the boundary.  |
| Kernal SVM | Use kernel trick to go to higher dimension i.e. use kernal methods which simplify calculation from low to high dim space |
| LDA | Supervised classification problem, find axes which maximize separation between means of classes and minimize the variance within the classes. |
|kNN| Supervised Classification and regression algo, average/mode of k closest data points. |

### Time Series

| Method             | Description        |
|--------------------|--------------------|
| Moving Average  |  Smoothing method, give equal importance to all data in the context window |
| Simple exponential smoothing  | Smoothing method, gives exponentially decay weight for moving average on neighboring data in context window. |
| Holt's exponential smoothing  | Double exponential smoothing, Can forecast predict level+trend |
| Winter-Holt's exponential smoothing  | Tripple exponential smoothing, can forecast level+trend+season, thus next cycle  |
| AR  | Forecast using lags  |
| MA  | Forecast using error/residual on previous lags |
|ARMA|Forecast using both lags (AR->PACF) and residual of previous lags (MA->ACF, Lunj-box test). Do Grid search to find AR and MA order which minimizes AIC the most while training|
| ARIMA |If the time series is not stationary. I in ARIMA stands which previous lags should be subtracted to remove the trend for the Time series to make it stationary.|
| SARIMA  | Same as ARIMA, except takes seasonality into account. Thus its additionally, seasonality AR, MA, I order.  |
| VAR | AR to forecast TS when forecast depends on lags from other correlated time series as well. Apply Gangar Causality to check correlation | 
|Hybrid models  | Use of feature transformers like linear regression to predict trend and target transformers like tree-based, NN to predict seasonality.  |
| Prophet | Curve fitting algo by Meta, can take weekly and yearly seasonality, holidays, and can have influx points for trend. |
| biLSTM | bi-directional LSTM module. Can handle multiple time series models as well to predict one TS. needs normalization but no stationarity of TS. |
| ARCH/GARCH  | Forecast using the residual ARCH, forecast using the error in the residual GARCH. Example, apply it in %change of stock price series or when you want to understand the residual of some time series when they exhibit trends/seasonality  |


### Clustering
| Method      | Type      | Description        |
|-------------|-------|--------------------|
| K-means | Hard, Centroid | Requires number of clusters prior (can be found via WCSS or Silhouette score), updates the centroid of each cluster iteratively.| 
| Generalised K means| Hard, Centroid | Can have clusters of different sizes and shapes, resistant to outliers | 
| K mediods| Hard, Centroid | Requires number of clusters prior. Uses actual data point as centroid| 
| Algomerative hierarchical | Hard, Hierarchial | Merge data points by closest distance iteratively up to some distance threshold.| 
| DBSCAN | Hard, Density | Does not require number of clusters prior, but min (radial) distance of a cluster (found by kNN distance plot) and number of points (default=2*features) | 
| Spectral | Hard | First step: move to the lower dimension by using PCA or graph-based dimension reduction (node as points, edge as distance). Then apply a clustering algorithm like Kmeans | 
| Fuzzy C means | Soft, Centroid | Update probability a point belongs to a centroid of a cluster iteratively. Needs number of clusters prior.| 
| Gaussian Mixture models | Soft, Distribution | Use Gaussian distribution to define clusters, optimal param of Gaussian distribution found using Expectation-Minimization method | 






## DL Papers

### Fully Convolution Neural Network

| Paper                                | Date       | Description                                      |
|--------------------------------------|------------|--------------------------------------------------|
| [LetNet-5](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=726791)        | 	December 1998 | CNN-AvgPool NN for softmax MNIST classifier, 1st usage of CNNs & weight sharing idea, activations: tanh, softmax  | 
| [AlexNet](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)                     | September 2012 | 1000 class classifier, Similar to LeNet but deeper, 1st usage of Dropout, ReLU    | 
| [VGGNet-16/19](https://arxiv.org/abs/1409.1556) | 	September 2014 | Very deep (16/19 layers) and narrow CNN, large number of small filters 3X3 to capture more complex and granular features, stacking of many layers  | 
| [InceptionNet](https://arxiv.org/abs/1409.4842) | 	September 2014 | Stacking of parallel modules, 1X1 followed by either 3X3 or 5X5, or 1X1 modules. 1X1 reduces channels dim & time (bottleneck layers) and diff nXn extracts low & high-level features. Global avg pooling counters overfitting & no. of params, Vanishing gradient countered by 2 auxiliary classifiers using same labels (0.3 weighted in final loss).                                              |
| [InceptionNetv2 & v3](https://arxiv.org/abs/1512.00567) |  December 2015   | Factorize nXn conv to 1Xn and nX1, thus improve speed. BN in auxiliary classifier, label smoothing for overfitting, RMSProp optimizer       |
| [ResNet](https://arxiv.org/abs/1512.03385)    |                       December 2015       | Residual connection in CNN blocks of deep CNN for vanishing grads.                                                               |
| [InceptionNetV4 and InceptionResNet](https://arxiv.org/abs/1602.07261) |   February 2016     | More uniform inception modules with new dim reduction blocks, skip connect module o/p to i/p (same channel dim achieved by 1X1) & replaced pooling operation. Scale residual activation to 0.3 to decrease vanishing grads.                              |
| [DenseNet](https://arxiv.org/abs/1608.06993)   |  August 2016   | Each layer in block receives input from all previous layers counter vanishing grads, transition layers between dense blocks reduce spatial dim to balance computational cost and accuracy.                                                             |
| [Xception](https://arxiv.org/abs/1610.02357) |   	October 2016      | Based on Inception v3, instead of inception modules,  it uses depthwise separable (depth + pointwise) convolutions over input image (grouped convolutions: i/p channel-wise conv, concatenate and apply pointwise 1X1 conv)                                                             |
| [ResNeXt](https://arxiv.org/abs/1608.06993)      | November 2016        | CNN with several Depthwise separable convolutions with ResNet.                                                            |
| [MobileNetv1](https://arxiv.org/abs/1704.04861)      | April 2017      | Depthwise separable convolutions network for mobile and embedded devices.                                                        |
| [MobileNetv2](https://arxiv.org/abs/1801.04381)      | 	January 2018    | [Inverted Residual](https://towardsdatascience.com/mobilenetv2-inverted-residuals-and-linear-bottlenecks-8a4362f4ffd5) connections (skip in narrow-wide-narrow type block with wide obtained by depthwise separable convolutions). Loss of extra info due to ReLU not coupled by wide-narrow-wide channel type block, linear activation in used in the final layer of block (linear bottleneck)                   |
| [MobileNetv3](https://arxiv.org/abs/1905.02244)  |  May 2019 | AutoML tools, MnasNet to select coarse architecture using reinforcement learning and NetAdapt to fine tune. Use [squeeze-and-excitation](https://medium.com/@tahasamavati/squeeze-and-excitation-explained-387b5981f249) block (per channel weights using global avg pooling and NN, apply to original channel), remove 3 expensive layers from v2.                                                              |
| [EfficientNet](https://arxiv.org/abs/1905.11946)  |May 2019      | AutoML automated neural architecture search (NAS) to select balanced dim of width, depth and resolution (compound scaling method). Usage of efficient building blocks like inverted residuals, linear bottleneck                   |

### Two Stage Object detectors
| Paper                                | Date       | Description                                      |
|--------------------------------------|------------|--------------------------------------------------|
| [RCNN](https://arxiv.org/abs/1311.2524) | 	November 2013 | Selective search for regions proposal (~2000), warp each RoI and pass each through CNN for feature extraction, linear SVM for classification and offset bbox regression. Non-max suppression to select from >1 bbox of same object                |
| [Fast RCNN](https://medium.com/dair-ai/papers-explained-15-fast-rcnn-28c1792dcee0)   | April 2015     | Pass entire image through large backbone CNN, selective search on output, crop, wrap each RoI and do RoI pooling and pass through small detection head NN for offset bbox regression and classification and objectness score. Faster inference due to usage of truncated SVD on model weights to retain imp weights/nodes from head. About 25X faster than RCNN.                            |
| [Faster RCNN](https://medium.com/dair-ai/papers-explained-16-faster-rcnn-a7b874ffacd9)   | June 2015        | Replace selective search in Fast RCNN by Region proposal neural network (RPN), 9 anchors different shape boxes to select best bbox per object.  About X10 faster than Fast RCNN       |
| [Mask RCNN](https://arxiv.org/abs/1703.06870)     | March 2017        | Add extra output to Faster RCNN to perform instance segmentation to classify each RoI.|
| [Cascade RCNN](https://arxiv.org/abs/1712.00726) | December 2017       | Detection head bbox output depends on RPN performance in Faster RCNN, coupled by poor inference conditional on IoU. To counter, cascade of individual RPN is used each trained using the preceding bbox RPN prediction trained for increasing IoU. |        

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
| [TinyBERT](https://arxiv.org/abs/1909.10351)    | 	September 2019     | Reduce Student BERT to 4 encoder block, 312 emb size (Teacher BERT has 12 blocks, 786 emb size). Knowledge distillation is not only at Prediction layer (as in DistilBERT), but also at the Embedding layer and transformer layer by minimizing MSE between them. Dimension mis-match b/w S and T solved by matrix factorization S(NX312) x Weight(312 X 768) = T(Nx768). Learn weight matrix also.                                             | 
| [GPT](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)    | June 2018       | Generative pre-training and discriminatory fine tuning. Architecture is similar to the decoder of original transformer. Unsupervised pre-training (thus able to use large corpus) using long Book texts (thus capture long range info) for standard language model objective of maximizing likelihood of sum(log P(xi|x1..x_i-1)). The pre-trained language model supervised fine tuned for text classification, text entailment x=(premise$hypothesis)->y=class, similarity x=(sent1$sent2) and x=(sent2$sent1)->y=1/0, question and answer mcq x(context$ans1), x(context$ans2)->softmax(2).   |
| [GPT-2](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)  | February 2019  | Demonstrated 0-shot learning (i.e. it learns language processing task without explicit supervision) with Large LM possible. Used new dataset WebText, modified BPE tokenizer, minor modification to decoder, context size 512->1024, 1.5B params. SOTA perplexity on language modelling (predict x_t|x1..x_t-1), accuracy on NER, perplexity/accuracy on LAMBDA data (predict last word of long range sent), accuracy on common sense reasoning, not SOTA ROUGE F1 in summarisation, BLUE in translation, comparable in accuracy in question-answer. These evaluation done without explicit training (fine tuning)|
| [GPT-3](https://arxiv.org/abs/2005.14165)  | May 2020  |  175B params, Auto-regressive language model. Similar model like GPT-2. SOTA on Language model, 35->20 on perplexity, accuracy score on 0 shot, one-shot, few shot; in question answer in 1/3 dataset over T5, not best for translation since 93% training in English, mixed for common sense reasoning, arithmetic task performance improves with shots |
| [BLOOM](https://arxiv.org/pdf/2211.05100.pdf)  |  November 2022 |  176B largest multilingual open source LL. Decoder only transformer similar to GPT-3, trained on 1.6TB hugging face dataset, 46 languages, 13 programming languages. Major architecture change using ALiBi Positional Embedding (attenuates attention score by the position info from distance between key and query, instead of addition position info to embedding layer), LayerNorm after embedding layer leading to better stability in training (effects zero-shot generalisation though). Pretrained, and the fine tuning for multitask and contrastive led to a multilingual information retrieval model and multilingual semantic textual similarity (STS) model. Performance is not effected on English-only task even though multilingual training. |


### Document AI

| Paper                                | Date       | Description                                    |            
|--------------------------------------|------------|--------------------------------------------------|
| [LayoutLM](https://arxiv.org/abs/1912.13318)    | December 2019        | Uses BERT as backbone model, takes trainable text embedding and new additional 4 positional embeddings (bbox of each token) to get layoutLM embedding. Text and bbox were extracted using pre-built OCR reader. At the same time, the ROI/bbox text in image form is passed through Faster-RCNN to get image embeddings. They are then used with LayoutLM emb. for fine-tuning tasks. Pre-trained for Masked Visual Language Model MLVM to learn text MSK using its position amb and other text+pos emb, Multi-label document classification to learn document representation in CLS token. Fine-tuned for form understanding task, receipt understanding task, and document image classification task i.e. text, layout info in pre-training and visual info in fine-tuning. WordPiece tokenizer, BIESO tags for token labeling.                      |
| [LayoutLMv2](https://arxiv.org/abs/2012.14740)    | December 2020        | Multi-modal emb with Spatial-Aware self-attention Mechanism (self-attention weights with bias vector to get relative spatial info). Pretraining using text, layout and image emb. Text emb = token emb + 1D position emb for token index + segment emb for different text segments. Visual emb = flatten features from ResNeXt-FPN + 1D position emb for token index + segment emb for different text segment, Layout emb - 6 bbox co-od. Pre-training: MVLM, Text image alignment TIA: cover images of token lines, predict covered or not, Text-Image matching: CLS token to predict if image is from the same text. Fine tune: Document image classification, token-level classification, visual question answering on document images.                                                                   |
| [LayoutLMv3]()    | April 2022        | ...                                                                   |
