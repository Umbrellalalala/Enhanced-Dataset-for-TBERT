import pandas as pd
import torch
import torch.nn as nn
from sent_att_model import SentAttNet
from word_att_model import WordAttNet
import json


class HierAttNet(nn.Module):
    def __init__(self, word_hidden_size, sent_hidden_size, batch_size, embed_table,
                 max_sent_length, max_word_length, connector_dict_path):
        """
        初始化HierAttNet模型。

        :param word_hidden_size: 单词级别隐藏层的大小
        :param sent_hidden_size: 句子级别隐藏层的大小
        :param batch_size: 批处理的大小
        :param embed_table: 词嵌入表
        :param max_sent_length: 最大句子长度
        :param max_word_length: 最大单词长度
        """
        super(HierAttNet, self).__init__()
        self.batch_size = batch_size
        self.word_hidden_size = word_hidden_size
        self.sent_hidden_size = sent_hidden_size
        self.max_sent_length = max_sent_length
        self.max_word_length = max_word_length
        #self.prompt_id = prompt_id

        # 初始化单词级别注意力网络
        self.word_att_net = WordAttNet(embed_table, word_hidden_size)
        # 初始化句子级别注意力网络
        self.sent_att_net = SentAttNet(sent_hidden_size, word_hidden_size)

        # 初始化隐藏状态
        self._init_hidden_state()
        """
        # 情感特征相关
        self.load_sentiment_data()
        self.sentiment_dict = {}
        self.polarity_weights = {}
        """
        # 加载衔接词字典
        self.load_connector_data(connector_dict_path)
        self.load_connector_weights()

        self.connector_weights = {}  # 初始化衔接词权重映射
    """
    def load_sentiment_data(self):
        
        # 读取情感词典和权重文件，并创建映射。
        
        # 读取情感词典
        senticnet_df = pd.read_excel("updated_senticnet.xlsx")
        # 读取权重文件
        ratios_df = pd.read_excel("essay_ratios.xlsx")

        # 创建情感值到权重的映射
        self.polarity_weights = {
            'negative': ratios_df['negative_ratio'].iloc[self.prompt_id - 1],
            'neutral': ratios_df['neutral_ratio'].iloc[self.prompt_id - 1],
            'positive': ratios_df['positive_ratio'].iloc[self.prompt_id - 1]
        }

        # 创建情感词典的映射
        self.sentiment_dict = {
            row['CONCEPT']: (row['POLARITY VALUE'], row['POLARITY INTENSITY'])
            for index, row in senticnet_df.iterrows()
        }
    """
    def load_connector_weights(self):
        """
        读取衔接词权重文件，并创建映射。
        """
        with open('connector_weights.json', 'r') as f:
            self.connector_weights = json.load(f)

    def load_connector_data(self, connector_dict_path):
        # 读取衔接词字典
        with open(connector_dict_path, 'r', encoding="utf-8") as f:
            self.connector_dict = json.load(f)

    def _init_hidden_state(self, last_batch_size=None):
        """
        初始化隐藏状态。如果提供了 last_batch_size，则使用该值来初始化。

        :param last_batch_size: 上一批的大小，默认使用当前批的大小
        """
        if last_batch_size:
            batch_size = last_batch_size
        else:
            batch_size = self.batch_size
        # 初始化单词级别和句子级别的隐藏状态
        self.word_hidden_state = torch.zeros(2, batch_size, self.word_hidden_size)
        self.sent_hidden_state = torch.zeros(2, batch_size, self.sent_hidden_size)
        # 如果可用，移动到GPU
        if torch.cuda.is_available():
            self.word_hidden_state = self.word_hidden_state.cuda()
            self.sent_hidden_state = self.sent_hidden_state.cuda()

    def forward(self, input):
        """
        前向传播函数。
        :param input: 输入数据，预期形状为 (batch_size, max_sent_length, max_word_length, input_dim)
        :return: 模型输出
        """
        # 初始化一个空列表来存储每个句子的输出
        outputs = []
        # 调整输入数据的维度
        input = input.permute(1, 0, 2)
        # 对每一个句子进行单词级别注意力计算
        for i in input:
            output = self.word_att_net(i.permute(1, 0))
            """
            # 考虑情感特征的权重
            for j in range(output.size(0)):
                # 获取单词的张量
                word_tensor = i[j]
                # 将张量转换为标量
                word = word_tensor.argmax().item()  # 选择最大值对应的索引作为单词的表示
                # 检查情感字典中是否存在该单词
                if word in self.sentiment_dict:
                    polarity_value, _ = self.sentiment_dict[word]
                    if polarity_value in self.polarity_weights:
                        weight = self.polarity_weights[polarity_value]
                        output[j] *= weight
            """

            # 融合衔接词特征
            for j in range(output.size(0)):
                word_tensor = i[j]
                word = word_tensor.argmax().item()
                if word in self.connector_dict:
                    connector_category = self.connector_dict[word]
                    if connector_category in self.connector_weights:
                        weight = self.connector_weights[connector_category]
                        output[j] *= weight

            outputs.append(output)

        output_list = torch.cat(outputs, dim=0)
        # 对整个句子集合进行句子级别注意力计算
        output = self.sent_att_net(output_list)
        return output
