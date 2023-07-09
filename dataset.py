from numpy import genfromtxt as LOAD_CSV_AS_ARRAY
from sklearn import datasets

def load_dataset(file_name):
    #data_set = LOAD_CSV_AS_ARRAY('./dataset/' + file_name, delimiter=",")
    iris = datasets.load_iris()
    data_set = iris.data  # we only take the first two features.
    labels = iris.target
    return data_set, len(data_set), labels


