import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class WordAttNet(nn.Module):
    def __init__(self, dict, hidden_size=100):
        """
        初始化 WordAttNet 模型。

        :param dict: 词嵌入词典，应该是一个 NumPy 数组
        :param hidden_size: 隐藏层的大小
        """
        super(WordAttNet, self).__init__()

        # 将词典从 NumPy 数组转换为 PyTorch tensor
        dict = torch.from_numpy(dict.astype(float))
        # 创建嵌入层，并加载预训练的词嵌入
        self.lookup = nn.Embedding(num_embeddings=4000, embedding_dim=50).from_pretrained(dict)

        # 定义卷积层，输入通道数为 50，输出通道数为 100，卷积核大小为 5
        self.conv1 = nn.Conv1d(in_channels=50, out_channels=100, kernel_size=5)

        # 定义 dropout 层，丢弃概率为 0.5
        self.dropout = nn.Dropout(p=0.5)

        # 定义全连接层
        self.fc1 = nn.Linear(100, 100)
        self.fc2 = nn.Linear(100, 1, bias=False)

    def forward(self, input):
        """
        前向传播函数。

        :param input: 输入数据，预期为词索引的 tensor
        :return: 模型输出
        """
        # 查找词嵌入
        output = self.lookup(input)

        # 应用 dropout
        output = self.dropout(output)

        # 调整维度，以匹配 Conv1d 的要求
        output = output.permute(1, 2, 0)

        # 应用卷积层
        f_output = self.conv1(output.float())  # 形状: batch * hidden_size * seq_len

        # 调整维度
        f_output = f_output.permute(2, 0, 1)  # 形状: seq_len * batch * hidden_size

        # 计算注意力权重
        weight = torch.tanh(self.fc1(f_output))
        weight = self.fc2(weight)
        weight = F.softmax(weight, 0)

        # 将注意力权重应用于卷积输出
        weight = weight * f_output
        output = weight.sum(0).unsqueeze(0)  # 形状: 1 * batch * hidden_size

        return output


if __name__ == "__main__":
    # 测试 WordAttNet 类
    abc = WordAttNet(np.random.rand(4000, 50))  # 这里使用随机数据作为示例
