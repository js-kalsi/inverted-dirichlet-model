from numpy import genfromtxt as LOAD_CSV_AS_ARRAY


def load_dataset(file_name):
    data_set = LOAD_CSV_AS_ARRAY('./dataset/' + file_name, delimiter=",")
    labels = data_set[:, -1]
    data_set = data_set[:, :-1]
    return data_set, len(data_set), labels


