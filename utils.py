import gzip
import logging
import sys
# from gensim.models.word2vec import Word2Vec
import numpy as np
import torch
from sympy.printing.theanocode import theano

# 定义ASAP评分范围的字典，每个prompt_id对应一个评分范围
asap_ranges = {
    0: (0, 60),
    1: (2, 12),
    2: (1, 6),
    3: (0, 3),
    4: (0, 3),
    5: (0, 4),
    6: (0, 4),
    7: (0, 30),
    8: (0, 60),
    11: (1, 3),
    12: (1, 3),
    13: (1, 3),
    14: (1, 3),
    15: (1, 3),
    16: (1, 3),
    17: (1, 3),
    18: (1, 3)
}


def convert_to_dataset_friendly_score(score, prompt_id):
    """将评分转换为数据集友好的分数格式"""
    low, high = asap_ranges[prompt_id]
    score = (score) * (high - low) + low  # 根据ASAP范围进行缩放
    return score


def get_logger(name, level=logging.INFO, handler=sys.stdout,
               formatter='%(name)s - %(levelname)s - %(message)s'):
    """设置日志记录器"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(formatter)
    stream_handler = logging.StreamHandler(handler)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def padding_sentence_sequences(index_sequences, scores, max_sentnum, max_sentlen, post_padding=True):
    """对句子序列进行填充，使其符合模型输入的尺寸要求"""
    X = np.empty([len(index_sequences), max_sentnum, max_sentlen], dtype=np.int32)
    Y = np.empty([len(index_sequences), 1], dtype=np.float32)
    mask = np.zeros([len(index_sequences), max_sentnum, max_sentlen])

    for i in range(len(index_sequences)):
        sequence_ids = index_sequences[i]
        num = len(sequence_ids)

        for j in range(num):
            word_ids = sequence_ids[j]
            length = len(word_ids)
            for k in range(length):
                wid = word_ids[k]
                X[i, j, k] = wid

            # 将序列末尾之后的X填充为0
            X[i, j, length:] = 0
            # 在掩码中，将有效长度范围内的值设为1
            mask[i, j, :length] = 1

        X[i, num:, :] = 0  # 对未使用的部分进行填充
        Y[i] = scores[i]

    return X, Y, mask


def padding_sequences(word_indices, char_indices, scores, max_sentnum, max_sentlen, maxcharlen, post_padding=True):
    """对单词和字符索引进行填充，支持字符特征"""
    X = np.empty([len(word_indices), max_sentnum, max_sentlen], dtype=np.int32)
    Y = np.empty([len(word_indices), 1], dtype=np.float32)
    mask = np.zeros([len(word_indices), max_sentnum, max_sentlen], dtype=theano.config.floatX)

    char_X = np.empty([len(char_indices), max_sentnum, max_sentlen, maxcharlen], dtype=np.int32)

    for i in range(len(word_indices)):
        sequence_ids = word_indices[i]
        num = len(sequence_ids)

        for j in range(num):
            word_ids = sequence_ids[j]
            length = len(word_ids)
            for k in range(length):
                wid = word_ids[k]
                X[i, j, k] = wid

            # 将序列末尾之后的X填充为0
            X[i, j, length:] = 0
            # 在掩码中，将有效长度范围内的值设为1
            mask[i, j, :length] = 1

        X[i, num:, :] = 0  # 对未使用的部分进行填充
        Y[i] = scores[i]

    # 对字符索引进行类似的填充
    for i in range(len(char_indices)):
        sequence_ids = char_indices[i]
        num = len(sequence_ids)
        for j in range(num):
            word_ids = sequence_ids[j]
            length = len(word_ids)
            for k in range(length):
                wid = word_ids[k]
                charlen = len(wid)
                for l in range(charlen):
                    cid = wid[l]
                    char_X[i, j, k, l] = cid
                char_X[i, j, k, charlen:] = 0
            char_X[i, j, length:, :] = 0
        char_X[i, num:, :] = 0

    return X, char_X, Y, mask


def load_word_embedding_dict(embedding, embedding_path, word_alphabet, logger, embedd_dim=100):
    """
    从文件加载词嵌入
    :param embedding: 嵌入类型（如glove, senna）
    :param embedding_path: 嵌入文件的路径
    :param logger: 日志记录器
    :return: 词嵌入字典, 嵌入维度, 是否大小写敏感
    """
    if embedding == 'glove':
        # 加载GloVe嵌入
        logger.info("Loading GloVe ...")
        embedd_dim = -1
        embedd_dict = dict()
        with open(embedding_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if len(line) == 0:
                    continue
                tokens = line.split()
                if embedd_dim < 0:
                    embedd_dim = len(tokens) - 1
                else:
                    assert (embedd_dim + 1 == len(tokens))
                embedd = np.empty([1, embedd_dim])
                embedd[:] = tokens[1:]
                embedd_dict[tokens[0]] = embedd
        return embedd_dict, embedd_dim, True
    elif embedding == 'senna':
        # 加载Senna嵌入
        logger.info("Loading Senna ...")
        embedd_dim = -1
        embedd_dict = dict()
        with gzip.open(embedding_path, 'r') as file:
            for line in file:
                line = line.strip()
                if len(line) == 0:
                    continue

                tokens = line.split()
                if embedd_dim < 0:
                    embedd_dim = len(tokens) - 1
                else:
                    assert (embedd_dim + 1 == len(tokens))
                embedd = np.empty([1, embedd_dim], dtype=theano.config.floatX)
                embedd[:] = tokens[1:]
                embedd_dict[tokens[0]] = embedd
        return embedd_dict, embedd_dim, True
    else:
        raise ValueError("embedding should choose from [glove, senna]")


def build_embedd_table(word_alphabet, embedd_dict, embedd_dim, logger, caseless):
    """构建词嵌入矩阵"""
    scale = np.sqrt(3.0 / embedd_dim)
    embedd_table = np.empty([len(word_alphabet), embedd_dim])
    embedd_table[0, :] = np.zeros([1, embedd_dim])  # 词表中的第一个词通常是填充词，对应全0向量
    oov_num = 0  # 未登录词的数量
    for word, index in word_alphabet.items():
        ww = word.lower() if caseless else word  # 根据是否大小写敏感来处理单词
        if ww in embedd_dict:
            embedd = embedd_dict[ww]
        else:
            embedd = np.random.uniform(-scale, scale, [1, embedd_dim])  # 对未登录词生成随机嵌入
            oov_num += 1
        embedd_table[index, :] = embedd
    oov_ratio = float(oov_num) / (len(word_alphabet) - 1)  # 计算未登录词比例
    logger.info("OOV number =%s, OOV ratio = %f" % (oov_num, oov_ratio))
    return embedd_table


def rescale_tointscore(scaled_scores, set_ids):
    '''
    将缩放后的分数重新缩放回原来的整数评分范围，基于给定的set_ids
    :param scaled_scores: 缩放后的分数，范围 [0,1]
    :param set_ids: 这些作文的对应set ID，取值范围为1到8的整数
    '''
    if isinstance(set_ids, int):
        prompt_id = set_ids
        set_ids = np.ones(scaled_scores.shape[0], ) * prompt_id  # 如果set_ids是单个值，将其转换为相应的数组
    assert scaled_scores.shape[0] == len(set_ids)  # 确保输入的分数和set_ids长度一致
    int_scores = np.zeros((scaled_scores.shape[0], 1))
    for k, i in enumerate(set_ids):
        assert i in range(1, 9)
        minscore = asap_ranges[i][0]
        maxscore = asap_ranges[i][1]
        int_scores[k] = scaled_scores[k] * (maxscore - minscore) + minscore  # 根据ASAP范围进行缩放

    return np.around(int_scores).astype(int)  # 将分数四舍五入为整数


def domain_specific_rescale(y_true, y_pred, set_ids):
    '''
    将分数重新缩放回原来的整数评分范围，并根据特定的prompt进行分区
    返回8个prompt的整数评分列表，分别为y_true和y_pred
    :param y_true: 真实分数列表，包含所有8个prompt的分数
    :param y_pred: 预测分数列表，也包含所有8个prompt的分数
    :param set_ids: 指示每篇作文的set/prompt ID的列表
    '''
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    y1_true, y1_pred = [], []
    y2_true, y2_pred = [], []
    y3_true, y3_pred = [], []
    y4_true, y4_pred = [], []
    y5_true, y5_pred = [], []
    y6_true, y6_pred = [], []
    y7_true, y7_pred = [], []
    y8_true, y8_pred = [], []

    for k, i in enumerate(set_ids):
        assert i in range(1, 9)
        minscore, maxscore = asap_ranges[i]
        y_true_score = y_true[k] * (maxscore - minscore) + minscore
        y_pred_score = y_pred[k] * (maxscore - minscore) + minscore
        if i == 1:
            y1_true.append(y_true_score)
            y1_pred.append(y_pred_score)
        elif i == 2:
            y2_true.append(y_true_score)
            y2_pred.append(y_pred_score)
        elif i == 3:
            y3_true.append(y_true_score)
            y3_pred.append(y_pred_score)
        elif i == 4:
            y4_true.append(y_true_score)
            y4_pred.append(y_pred_score)
        elif i == 5:
            y5_true.append(y_true_score)
            y5_pred.append(y_pred_score)
        elif i == 6:
            y6_true.append(y_true_score)
            y6_pred.append(y_pred_score)
        elif i == 7:
            y7_true.append(y_true_score)
            y7_pred.append(y_pred_score)
        elif i == 8:
            y8_true.append(y_true_score)
            y8_pred.append(y_pred_score)
        else:
            print("Set ID error")

    prompts_truescores = [np.around(y1_true), np.around(y2_true), np.around(y3_true), np.around(y4_true),
                          np.around(y5_true), np.around(y6_true), np.around(y7_true), np.around(y8_true)]
    prompts_predscores = [np.around(y1_pred), np.around(y2_pred), np.around(y3_pred), np.around(y4_pred),
                          np.around(y5_pred), np.around(y6_pred), np.around(y7_pred), np.around(y8_pred)]

    return prompts_truescores, prompts_predscores
