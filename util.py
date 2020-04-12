#将输入结果进行归一化
def normalise_windows(window_data):
    normalised_data = []
    for window in window_data:   #window shape (sequence_length L ,)  即(51L,)
        normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
        normalised_data.append(normalised_window)
    return normalised_data

#将预测的结果进行反归一化，获得真实预测结果
def FNormalise_windows(window,data):
    normalised_data = []
    for i in range(len(window)):   #window shape (sequence_length L ,)  即(51L,)
        normalised_data.append((float(data[i]) + 1 ) * float(window[i]))
    return normalised_data