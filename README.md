connective：衔接词

sentiment：情感

# CNN-LSTM-ATT model for essay scoring

This is a Pytorch implementations for paper Attention-based Recurrent Convolutional Neural Network for Automatic Essay Scoring.
[[pdf](https://www.aclweb.org/anthology/K17-1017.pdf)]

# Version
Our version is:
- Python 3.6
- PyTorch 1.8.0

# Training
python train.py --oov embedding --embedding glove --embedding_dict glove.6B.50d.txt --embedding_dim 50 --datapath data/fold_ --prompt_id 1
// oov:OOV词汇是指在训练词汇表中未见过的词汇。这里选择的是用embedding来处理OOV词汇，可能表示将这些词汇映射到某种词向量表示。
// embedding: 选择的词向量表示方法，这里选择的是glove。
// embedding_dict: 词向量表示的字典，这里选择的是glove.6B.50d.txt。
// embedding_dim: 词向量的维度，这里选择的是50。
// datapath: 数据集的路径，这里选择的data/fold_。
// prompt_id: 作文题目的id，这里选择的是1。

Note that you should download glove.6B.50d.txt.

GloVe是一个用于获取词向量表示的非监督学习算法。训练过程基于语料库中词与词的共现统计信息，通过汇总全局的共现信息来进行。学习到的词向量展现了词向量空间中有趣的线性子结构。
GloVe模型在各种自然语言处理任务中都取得了很好的效果，例如文本分类、语义分析、机器翻译等。

# 项目结构
- data: 数据集，内含train、dev、test。
train (训练集): 这是用于训练模型的数据集。模型会通过这个数据集来学习特征和模式。
dev (开发集/验证集): 这是用于调优模型超参数和评估模型性能的数据集。开发集帮助确定模型是否过拟合以及选择最优模型。
test (测试集): 这是用于最终评估模型泛化能力的数据集。测试集的数据没有参与训练和模型调优，反映模型在真实世界中的表现。
- src: 代码文件
    - utils.py: 一些工具函数
- data_prepare.py: 数据预处理
- glove.6B.50d.txt: glove词向量
- hierarchical_att_model.py: 模型文件
- metrics.py: 评价指标
- reader.py: 读取数据
- README.md: 项目说明
- sent_att_model.py: 模型文件
- train.py: 训练文件
- utils.py: 工具函数
- word_att_model.py: 模型文件

# 阅读代码流程
数据读取与处理：data -> reader.py -> data_prepare.py
模型与演变：hierarchical_att_model.py -> word_att_model.py -> sent_att_model.py
训练与评判：train.py -> metrics.py