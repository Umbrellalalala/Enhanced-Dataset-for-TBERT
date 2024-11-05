import torch
import torch.nn as nn
import torch.nn.functional as F


class SentAttNet(nn.Module):
    def __init__(self, sent_hidden_size=100, word_hidden_size=100):
        """
        初始化 SentAttNet 模型。

        :param sent_hidden_size: LSTM 隐藏层的大小
        :param word_hidden_size: 输入词向量的大小
        """
        super(SentAttNet, self).__init__()

        # 定义 LSTM 层
        self.LSTM = nn.LSTM(word_hidden_size, sent_hidden_size)

        # 定义全连接层，用于将 LSTM 输出映射到一个标量
        self.fc = nn.Linear(sent_hidden_size, 1)

        # 定义全连接层，用于计算注意力权重
        self.fc1 = nn.Linear(sent_hidden_size, sent_hidden_size)
        self.fc2 = nn.Linear(sent_hidden_size, 1, bias=False)

    def forward(self, input):
        """
        前向传播函数。

        :param input: 输入数据，应该是 LSTM 需要的格式
        :return: 模型输出
        """
        # 通过 LSTM 层
        f_output, _ = self.LSTM(input)

        # 计算注意力权重
        weight = torch.tanh(self.fc1(f_output))
        weight = self.fc2(weight)
        weight = F.softmax(weight, dim=0)

        # 将注意力权重应用于 LSTM 输出
        weight = weight * f_output
        output = weight.sum(0)

        # 通过全连接层输出结果，并使用 sigmoid 函数进行归一化
        output = torch.sigmoid(self.fc(output))

        return output


if __name__ == "__main__":
    # 测试 SentAttNet 类
    abc = SentAttNet()
