import numpy as np


def ecogGetDefC(TRAIN):
    """
    根据Joachims方法计算SVM的默认C参数

    参数:
        TRAIN: 训练数据矩阵 (样本数 x 特征数)

    返回:
        C: 计算得到的默认参数值
    """
    # 计算每个样本与自身的点积（即L2范数的平方）
    scalar_prod = np.sum(TRAIN * TRAIN, axis=1)

    # 计算平均点积的倒数作为C值
    C = 1.0 / np.mean(scalar_prod)

    return C