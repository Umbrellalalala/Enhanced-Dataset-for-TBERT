import reader
import utils
import numpy as np

# 设置日志记录器，用于输出数据准备阶段的日志
logger = utils.get_logger("Prepare data ...")


def prepare_sentence_data(datapaths, embedding_path=None, embedding='word2vec', embedd_dim=100, prompt_id=1,
                          vocab_size=0, tokenize_text=True, \
                          to_lower=True, sort_by_len=False, vocab_path=None, score_index=6):
    """
    准备句子数据，包括加载数据、填充序列、缩放分数等步骤。

    :param datapaths: 包含训练、验证和测试数据路径的列表
    :param embedding_path: 词嵌入文件的路径
    :param embedding: 词嵌入的类型（如 'word2vec'）
    :param embedd_dim: 词嵌入的维度
    :param prompt_id: 评分的提示ID
    :param vocab_size: 词汇表大小
    :param tokenize_text: 是否进行分词
    :param to_lower: 是否将文本转换为小写
    :param sort_by_len: 是否按长度排序
    :param vocab_path: 词汇表文件路径
    :param score_index: 分数的索引位置
    :return: 训练、验证和测试数据的元组，以及词汇表、词嵌入矩阵等
    """

    # 确保提供了训练、验证和测试数据路径
    assert len(datapaths) == 3, "data paths should include train, dev and test path"

    # 从文件中读取数据
    (train_x, train_y, train_prompts), (dev_x, dev_y, dev_prompts), (
    test_x, test_y, test_prompts), vocab, overal_maxlen, overal_maxnum = \
        reader.get_data(datapaths, prompt_id, vocab_size, tokenize_text, to_lower, sort_by_len, vocab_path, score_index)

    # 对训练、验证和测试数据进行填充
    X_train, y_train, mask_train = utils.padding_sentence_sequences(train_x, train_y, overal_maxnum, overal_maxlen,
                                                                    post_padding=True)
    X_dev, y_dev, mask_dev = utils.padding_sentence_sequences(dev_x, dev_y, overal_maxnum, overal_maxlen,
                                                              post_padding=True)
    X_test, y_test, mask_test = utils.padding_sentence_sequences(test_x, test_y, overal_maxnum, overal_maxlen,
                                                                 post_padding=True)

    # 如果提供了 prompt_id，则转换成 numpy 数组
    if prompt_id:
        train_pmt = np.array(train_prompts, dtype='int32')
        dev_pmt = np.array(dev_prompts, dtype='int32')
        test_pmt = np.array(test_prompts, dtype='int32')

    # 计算训练、验证和测试数据的均值和标准差
    train_mean = y_train.mean(axis=0)
    train_std = y_train.std(axis=0)
    dev_mean = y_dev.mean(axis=0)
    dev_std = y_dev.std(axis=0)
    test_mean = y_test.mean(axis=0)
    test_std = y_test.std(axis=0)

    # 将分数转换为模型友好的格式，并计算训练数据的均值
    Y_train = reader.get_model_friendly_scores(y_train, train_prompts)
    Y_dev = np.array(y_dev)
    Y_test = np.array(y_test)
    scaled_train_mean = Y_train.mean(axis=0)

    # 输出数据统计信息
    logger.info('Statistics:')
    logger.info('  train X shape: ' + str(X_train.shape))
    logger.info('  dev X shape:   ' + str(X_dev.shape))
    logger.info('  test X shape:  ' + str(X_test.shape))
    logger.info('  train Y shape: ' + str(Y_train.shape))
    logger.info('  dev Y shape:   ' + str(Y_dev.shape))
    logger.info('  test Y shape:  ' + str(Y_test.shape))
    logger.info('  train_y mean: %s, stdev: %s, train_y mean after scaling: %s' %
                (str(train_mean), str(train_std), str(scaled_train_mean)))

    # 如果提供了嵌入路径，则加载词嵌入并构建嵌入矩阵
    if embedding_path:
        embedd_dict, embedd_dim, _ = utils.load_word_embedding_dict(embedding, embedding_path, vocab, logger,
                                                                    embedd_dim)
        embedd_matrix = utils.build_embedd_table(vocab, embedd_dict, embedd_dim, logger, caseless=True)
    else:
        embedd_matrix = None

    # 返回处理后的数据、词汇表及其他信息
    return (X_train, Y_train, mask_train, train_prompts), (X_dev, Y_dev, mask_dev, dev_prompts), (
    X_test, Y_test, mask_test, test_prompts), \
           vocab, len(vocab), embedd_matrix, overal_maxlen, overal_maxnum, scaled_train_mean
