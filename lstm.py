from keras.models import Sequential
from keras.layers import Bidirectional,LSTM
from keras.layers.core import Dense, Activation, Dropout
import time
from contextlib import redirect_stdout


def lstm():
    # 接下来，构建一个LSTM神经网络模型。通过add来增加一层神经网络；加入一层输出为1维的神经网络，
    # 并设置激活层函数为线性（选择线性的原因，是因为我这里试用了多种激活函数，发现线性激活函数效果最好）
    # 最终，对模型进行编译，回归问题，
    # 损失函数为mse，优化器选择rmsprop。
    model = Sequential()
    # layers [1,50,50,50,50,1]
    # input_shape = (None,1)中的None代表模型输入集的数据量不限制（即可以是过去任意天的工作量），
    # 1在这里代表只有工作量一个维度。
    # units = 50 代表 代表将输入的维度映射成50个维度输出。
    # return_sequences为True意味着返回多个单元短期的输出结果,为False则只返回一个单元的输出结果。
    model.add(LSTM(input_shape=(None, 1), units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(input_shape=(None, 50), units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(input_shape=(None, 50), units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.add(Activation("linear"))
    # model.add(Activation("sigmoid"))
    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop")
    print("Compilation Time : ", time.time() - start)
    print("LSTM层数：", len(model.layers))
    with open('E:\C-工作内容\\0-策略小组\AI\model_summary.txt', 'w+') as f:
        with redirect_stdout(f):
            model.summary()
        f.close()
    return model



