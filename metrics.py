#coding = utf-8
import numpy as np

def pearson(pred_score, true_score):
    # 计算预测值和真实值的均值
    pred_avg = np.average(pred_score)
    true_avg = np.average(true_score)

    num, n1, n2 = 0.0, 0.0, 0.0
    # 计算协方差和方差
    for pred_t, true_t in zip(pred_score, true_score):
        num += (pred_t - pred_avg) * (true_t - true_avg)
        n1 += (pred_t - pred_avg) * (pred_t - pred_avg)
        n2 += (true_t - true_avg) * (true_t - true_avg)

    # 计算 Pearson 相关系数
    return num / np.power(n1 * n2, 0.5)



def spearman(pred_score, true_score):
    pred_score = np.asarray(pred_score)
    true_score = np.asarray(true_score)

    # 对预测值和真实值进行排序
    pred_sort = np.sort(pred_score)
    true_sort = np.sort(true_score)

    pred_index, true_index = [], []
    # 计算排序后的秩
    for pred_t, true_t in zip(pred_score, true_score):
        index_list = np.where(pred_sort == pred_t)
        index = (index_list[0] + index_list[-1]) / 2
        pred_index.append(index[0])

        index_list = np.where(true_sort == true_t)
        index = (index_list[0] + index_list[-1]) / 2
        true_index.append(index[0])

    nb = len(pred_score)
    err = 0.0
    # 计算秩差的平方和
    for pred_i, true_i in zip(pred_index, true_index):
        err += np.power(pred_i - true_i, 2)

    # 计算 Spearman 秩相关系数
    return 1.0 - 6.0 * err / (np.power(nb, 3) - nb)


import numpy as np


def confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):
    assert (len(rater_a) == len(rater_b))

    # 将评分者的评分转换为一维数组
    rater_a = np.array(rater_a).flatten()
    rater_b = np.array(rater_b).flatten()

    # 如果没有指定最小值和最大值，则自动计算
    if min_rating is None:
        min_rating = min(np.min(rater_a), np.min(rater_b)).item()
    if max_rating is None:
        max_rating = max(np.max(rater_a), np.max(rater_b)).item()

    num_ratings = int(max_rating - min_rating + 1)
    # 创建混淆矩阵
    conf_mat = [[0 for _ in range(num_ratings)] for _ in range(num_ratings)]

    # 填充混淆矩阵
    for a, b in zip(rater_a, rater_b):
        a = int(a)  # 确保是整数
        b = int(b)  # 确保是整数
        max_rating = int(max_rating)
        min_rating = int(min_rating)
        conf_mat[a - min_rating][b - min_rating] += 1

    return conf_mat

def histogram(ratings, min_rating=None, max_rating=None):
    # 计算评分的直方图
    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
        r = int(r)
        min_rating = int(min_rating)
        hist_ratings[r - min_rating] += 1
    return hist_ratings

def quadratic_weighted_kappa(rater_a, rater_b, min_rating=None, max_rating=None):
    # 将评分者的评分转换为整数类型
    rater_a = np.array(rater_a, dtype=int)
    rater_b = np.array(rater_b, dtype=int)
    assert(len(rater_a) == len(rater_b))

    # 如果没有指定最小值和最大值，则自动计算
    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))
    # 计算混淆矩阵和直方图
    conf_mat = confusion_matrix(rater_a, rater_b, min_rating, max_rating)

    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    # 计算加权的平方差
    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j] / num_scored_items)
            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    # 计算 Quadratic Weighted Kappa
    return 1.0 - numerator / denominator

