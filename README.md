# AI_in_GIST

**Fully Convolution Neural Network**

| Paper                                | Date       | Description                                      | Task                     |
|--------------------------------------|------------|--------------------------------------------------|--------------------------|
| [LetNet-5](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=726791)        | 1988 | CNN-AvgPool NN for softmax MNIST classifier, 1st usage of CNNs & weight sharing idea, activations: tanh, softmax  | Multiclass Classification  |
| [AlexNet](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)                     | 2012 | 1000 class classifier, Similar to LeNet but deeper, 1st usage of Dropout, ReLU    | Classification  |
| [VGGNet-16/19](https://arxiv.org/abs/1409.1556) | 2014 | Very deep (16/19 layers) and narrow CNN, large number of small filters 3X3 to capture more complex and granular features, stacking of many layers  | Classification  |
| [InceptionNet](https://arxiv.org/abs/1409.4842) | 2014 | Stacking of parallel modules, 1X1 followed by either 3X3 or 5X5, or 1X1 modules. 1X1 reduces channels dim & time (bottleneck layers) and diff nXn extracts low & high-level features. Global avg pooling counters overfitting & no. of params, Vanishing gradient countered by 2 auxiliary classifiers using same labels (0.3 weighted in final loss).                                              | Classification                      |
| [InceptionNetv2 & v3](https://arxiv.org/abs/1512.00567) |     | Factorize nXn conv to 1Xn and nX1, thus improve speed. BN in auxiliary classifier, label smoothing for overfitting, RMSProp optimizer       | ...                                              | 
| [ResNet](https://arxiv.org/abs/1512.03385)    |                              | Residual connection in CNN blocks of deep CNN for vanishing grads.                                                | ...                      |
| [InceptionNetV4 and InceptionResNet](https://arxiv.org/abs/1602.07261) |        | More uniform inception modules with new dim reduction blocks, skip connect module o/p to i/p (same channel dim achieved by 1X1) & replaced pooling operation. Scale residual activation to 0.3 to decrease vanishing grads.  | ...                               |
| [DenseNet](https://arxiv.org/abs/1608.06993)   |     | Each layer in block receives input from all previous layers counter vanishing grads, transition layers between dense blocks reduce spatial dim to balance computational cost and accuracy.                                            | ...                      |
| [Xception](https://arxiv.org/abs/1610.02357) |   2017      | Based on Inception v3, instead of inception modules,  it uses depthwise separable (depth + pointwise) convolutions over input image (grouped convolutions: i/p channel-wise conv, concatenate and apply pointwise 1X1 conv)                                              | ...                      |
| [ResNeXt](https://arxiv.org/abs/1608.06993)      | 2016        | CNN with several Depthwise separable convolutions with ResNet.                                            | ...                      |
| [MobileNetv1](https://arxiv.org/abs/1704.04861)      | 2017       | Depthwise separable convolutions network for mobile and embedded devices.                                           | ...                      |
| [MobileNetv2](https://arxiv.org/abs/1801.04381)      | 2018    | Inverted Residual connections (skip in narrow-wide-narrow type block with wide obtained by depthwise separable convolutions). Loss of extra info due to ReLU not coupled by wide-narrow-wide channel type block, linear activation in used in the final layer of block (linear bottleneck)     | ...                      |
| [MobileNetv3](https://arxiv.org/abs/1905.02244)   | ...        | AutoML tools, MnasNet to select coarse architecture using reinforcement learning and NetAdapt to fine tune. Use squeeze-and-excitation block (per channel weights using global avg pooling and NN, apply to original channel), remove 3 expensive layers from v2.                                            | ...                      |
| [EfficientNet](https://arxiv.org/abs/1905.11946)  |2020      | AutoML automated neural architecture search (NAS) to select balanced dim of width, depth and resolution (compound scaling method). Usage of efficient building blocks like inverted residuals, linear bottleneck    | ...                      |

**Two Stage Object detectors**
| Paper                                | Date       | Description                                      | Task                     |
|--------------------------------------|------------|--------------------------------------------------|--------------------------|
| [RCNN](https://arxiv.org/abs/1311.2524)   | Selective search for regions proposal (~2000), warp each RoI and pass each through CNN for feature extraction, linear SVM for classification and offset bbox regression. Non-max suppression to select from >1 bbox of same object        | ...                          ...            |
| Fast RCNN   | ...        | Pass entire image through large backbone CNN, selective search on output, crop, wrap each RoI and do RoI pooling and pass through small detection head NN for offset bbox regression and classification and objectness score. Faster inference due to usage of truncated SVD on model weights to retain imp weights/nodes from head. About 25X faster than RCNN.                         ...            |
| Faster RCNN   | ...        | Replace selective search in Fast RCNN by Region proposal neural network (RPN), 9 anchors different shape boxes to select best bbox per object.  About X10 faster than Fast RCNN                      ...            |
| [Mask RCNN](https://arxiv.org/abs/1703.06870)     | ...        | Add extra output to Faster RCNN to perform instance segmentation to classify each RoI.                          ...            |
| [Cascade RCNN](https://arxiv.org/abs/1712.00726) | ...        | Detection head bbox output depends on RPN performance in Faster RCNN, coupled by poor inference conditional on IoU. To counter, cascase of individual RPN are used each trained using preceeding bbox RPN prediction trained for increasing IoU. |           ...            |


**Vision Transformers**
* Generally requires more data than CNN, 

| Paper                                | Date       | Description                                      | Task                     |
|--------------------------------------|------------|--------------------------------------------------|--------------------------|
| [Vision Transformer]()    | ...        | Requires more data than CNN,                                              | ...                      |
| ...    | ...        | ...                                              | ...                      |
| ...    | ...        | ...                                              | ...                      |
| ...    | ...        | ...                                              | ...                      |
| ...    | ...        | ...                                              | ...                      |

**NLP/Language Models**

| Paper                                | Date       | Description                                      | Task                     |
|--------------------------------------|------------|--------------------------------------------------|--------------------------|
| [Transformer](https://arxiv.org/abs/1706.03762)    | 	June 2017   | Encoder-decoder model, Multi-head self-attention (Query, Key, Value) in encoder and Masked multihead attention MLM and multi-head cross attention (K,V from encoder output, V from decoder), skip connections, [Layer normalisation](https://stats.stackexchange.com/questions/474440/why-do-transformers-use-layer-norm-instead-of-batch-norm) (across all features of a token), BPE used as tokenizer, trainable text 512 length embedding, fixed non-trainable absolute position encoding (sin/cos), trained on WMT English-German, English-Frech dataset.                                              | ...                      |
| [BERT](https://arxiv.org/abs/1810.04805)    | October 2018 | Encoder transformer model, max input 512 tokens, WordPiece tokenizer, trainable text embedding length 768, fixed non-trainable absolute position encoding (sin/cos), multi-head self-attention (Query, Key, Value), final linear-relu-softmax layer, trained on left-right context (bidirectional embedding), Pre-training (MLM task (MASK token), Next Sentence Prediction NSP task using CLS-sent1-SEP-sent2-SEP tokens), BERT fine-tuning (Text Classification CLS token, Question-Answer Task CLS-ques-SEP-context-SEP).                 |
| [RoBERTa](https://arxiv.org/abs/1907.11692)    | July 2019        | Optimize BERT, Dynamic masking of tokens in MLM in training pre-training BERT, no NSP due to low value, ByteEncoder, X10 train data (160GB), larger batch size                                              | ...                      |
| [ELECTRA]()    |         | Replaced Token Detection instead of MLM (Generator BERT predicts MSK tokens, Discriminator BERT predicts isOriginal or isReplaced) thus focus on all tokens in sequence and not just MSK token in BERT, no NSP                                               | ...                      |
| [ALBERT]()    | Cross param sharing across BERT blocks, Embedding matrix factorization (AXB=(AXN)*(NXB)) leading to 1/10 size of BERT. Pre-training MLM and Sentence order prediction SOP, no NSP since it makes more sense to know inter-sentence coherence.        |                                               | ...                      |
| [DistilBERT]()   | ...        | 60X faster, 0.40 size, 97% BERT accurate. Teacher-student knowledge distillation. Train: train teacher BERT for MLM task. Student BERT soft prediction with teacher BERT soft target using KL divergence loss, student BERT hard prediction with hard target using CE loss and cosine similarity loss between soft target and soft prediction embeddings.                                             | ...                      |
| ...    | ...        | ...                                              | ...                      |
| ...    | ...        | ...                                              | ...                      |
