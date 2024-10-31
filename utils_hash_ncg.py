# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
import numpy as np
from tqdm import tqdm

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None


def mean_average_precision_R_return(database_hash, test_hash, database_labels, test_labels, R, num_classes):

    one_hot_database = np.eye(num_classes)[database_labels.astype(int)]
    one_hot_test = np.eye(num_classes)[test_labels.astype(int)]

    if R == -1:
        R = database_hash.shape[0]
    query_num = test_hash.shape[0]  # total number for testing
    sim = np.dot(database_hash, test_hash.T)
    ids = np.argsort(-sim, axis=0)

    APx = []
    top30_dindices = []

    for i in tqdm(range(query_num)):  # for i=0
        label = one_hot_test[i, :]  # the first test labels
        if np.sum(label) == 0:  # ignore images with meaningless label in nus wide
            continue
        label[label == 0] = -1
        idx = ids[:, i]
        imatch_acg = np.sum(one_hot_database[idx[0:R], :] == label, axis=1)
        imatch = imatch_acg > 0
        relevant_num = np.sum(imatch)
        Lx = np.cumsum(imatch)   # 累加
        Px = Lx.astype(float) / np.arange(1, R + 1, 1)

        if relevant_num != 0:
            APx.append(np.sum(Px * imatch) / relevant_num)
            # wAPx.append(np.sum(imatch * P_acg) / relevant_num)
        if relevant_num == 0:  # even no relevant image, still need add in APx for calculating the mean
            APx.append(0)

        top30_dindices.append(idx[:30])

    return np.mean(np.array(APx)), top30_dindices


def mean_average_precision_R(database_hash, test_hash, database_labels, test_labels, R, num_classes):

    one_hot_database = np.eye(num_classes)[database_labels.astype(int)]
    one_hot_test = np.eye(num_classes)[test_labels.astype(int)]

    if R == -1:
        R = database_hash.shape[0]
    query_num = test_hash.shape[0]  # total number for testing
    sim = np.dot(database_hash, test_hash.T)
    ids = np.argsort(-sim, axis=0)

    APx = []
    Recall = []
    # wAPx = []

    for i in tqdm(range(query_num)):  # for i=0
        label = one_hot_test[i, :]  # the first test labels
        if np.sum(label) == 0:  # ignore images with meaningless label in nus wide
            continue
        label[label == 0] = -1
        idx = ids[:, i]
        imatch_acg = np.sum(one_hot_database[idx[0:R], :] == label, axis=1)
        imatch = imatch_acg > 0
        relevant_num = np.sum(imatch)
        Lx = np.cumsum(imatch)   # 累加
        Px = Lx.astype(float) / np.arange(1, R + 1, 1)

        if relevant_num != 0:
            APx.append(np.sum(Px * imatch) / relevant_num)
            # wAPx.append(np.sum(imatch * P_acg) / relevant_num)
        if relevant_num == 0:  # even no relevant image, still need add in APx for calculating the mean
            APx.append(0)
            # wAPx.append(0)
        # print(i)

        all_relevant = np.sum(one_hot_database == label, axis=1) > 0
        all_num = np.sum(all_relevant)
        r = relevant_num / np.float64(all_num)
        Recall.append(r)
    return np.mean(np.array(APx)), np.mean(np.array(Recall))

# 标签对应关系映射
label_mapping = {
     25: [19],
     19: [25],
     99: [5],
     5: [99],
     45: [50],
     50: [45],
     84: [70],
     70: [84],
     78: [59],
     59: [78],
     97: [66],
     66: [97],
     71: [14],
     14: [71],
     63: [41],
     41: [63],
     73: [6],
     6: [73]
}


def acg_test(database_hash, test_hash, database_labels, test_labels, n):
    print(test_labels)
    query_num = test_hash.shape[0]  # total number for testing
    sim = np.dot(database_hash, test_hash.T)
    ids = np.argsort(-sim, axis=0)

    ACG_all = []
    NDCG_all = []

    for i in range(query_num):  # for i=0
        label = test_labels[i]  # 获取当前测试样本的标签
        idx = ids[:, i]
        imatch = []
        imatch_all = []

        for j in range(n):
            db_label = database_labels[idx[j]]
            if db_label == label:
                imatch.append(2)
            elif db_label in label_mapping.get(label, []):
                imatch.append(1)
            else:
                imatch.append(0)

        for db_label in database_labels:
            if db_label == label:
                imatch_all.append(2)
            elif db_label in label_mapping.get(label, []):
                imatch_all.append(1)
            else:
                imatch_all.append(0)

        ACG_all.append(np.mean(imatch))

        DCG = (2 ** np.array(imatch) - 1) / np.log2(np.arange(2, n + 2, 1))
        DCG = np.sum(DCG)

        imatch_sort = np.array(sorted(imatch_all, reverse=True))
        DCG_max = np.sum((2 ** imatch_sort[:n] - 1) / np.log2(np.arange(2, n + 2, 1)))
        
        NDCG_all.append(DCG / DCG_max)

    return np.mean(np.array(ACG_all)), np.mean(np.array(NDCG_all))
