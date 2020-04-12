import time
import warnings
import numpy as np
from numpy import newaxis
import pandas as pd
import util

# 划分训练集、测试集
def load_data(filename, seq_len, normalise_window):
    data = open(filename).readlines()
    sequence_length = seq_len + 2
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])
        # 得到长度为seq_len+1的向量，最后一个作为label

    # 记录normalise_windows归一化前，x_test中window[0]值
    re = np.array(result).astype('float32')
    print(type(result[0][0]))
    # print('re is:',re)
    ro = int(round(0.9 * len(result)))  # 划分训练集和测试集
    # print('ro is:',ro)
    window = re[ro:, :1]
    print('window is:', window)
    # normalise_windows归一化
    if normalise_window:
        result = util.normalise_windows(result)
        # 以最初日期的数据进行初始化
    # print('result is:',result)
    result = np.array(result)

    # 划分train、test
    row = round(0.9 * result.shape[0])
    row = int(row)

    train = result[:row, :]

    np.random.shuffle(train)
    x_train = train[:, :-1]
    y_train = train[:, -1:]
    x_test = result[row:, :-1]
    y_test = result[row:, -1:]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    return [x_train, y_train, x_test, y_test, window]
    # window是最终的测试结果的真实值


if __name__ == '__main__':
    X_train, y_train, X_test, y_test, window \
        = load_data('E:\C-工作内容\\0-策略小组\AI\data.csv', 50, True)
    print('X_train shape:', X_train.shape)
    print('y_train shape:', y_train.shape)
    print('X_test shape:', X_test.shape)
    print('y_test shape:', y_test.shape)