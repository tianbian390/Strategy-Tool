import data
import numpy as np
import util
import matplotlib.pyplot as plt
import pandas as pd
import lstm
import time


def predict_point_by_point(model, data):
    predicted = model.predict(data)
    print('predicted shape:', np.array(predicted).shape)
    predicted = np.reshape(predicted, (predicted.size,))
    return predicted


def plot_results(predicted_data, true_data, filename):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()
    plt.savefig(filename+'.png')



if __name__ == '__main__':
    import importlib, sys
    importlib.reload(sys)
    df = pd.read_csv('E:\C-工作内容\\0-策略小组\AI\\test-2.csv')
    df.plot()
    plt.show()
    print(df.shape)

    X_train, y_train, X_test, y_test, window \
        = data.load_data('E:\C-工作内容\\0-策略小组\AI\\test-2.csv', 119, True)
    print('X_train shape:', X_train.shape)
    print('y_train shape:', y_train.shape)
    print('X_test shape:', X_test.shape)
    print('y_test shape:', y_test.shape)

    model = lstm.lstm()
    start = time.time()
    print("Begin to fit:",  time.asctime(time.localtime(start)))
    model.fit(X_train, y_train, batch_size=10, nb_epoch=100, validation_split=0.05,
              verbose=0)
    close = time.time()
    print("Finish:", time.asctime(time.localtime(close)), "\nFit time", close - start)
    score2 = model.evaluate(X_test, y_test)
    print(score2)

    point_by_point_predictions = predict_point_by_point(model, X_test)
    point_by_point_predictions = util.FNormalise_windows(window, point_by_point_predictions)

    y_test = util.FNormalise_windows(window, y_test)
    plt.scatter(range(len(point_by_point_predictions)), y_test, color='b', s=5)
    plt.scatter(range(len(point_by_point_predictions)), point_by_point_predictions,
                s=5, color='r', marker='o')
    plot_results(point_by_point_predictions, y_test, 'point_by_point_predictions')
    plt.show()