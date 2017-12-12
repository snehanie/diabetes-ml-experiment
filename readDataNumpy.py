from numpy import loadtxt
from urllib import urlopen
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data'
raw_data = urlopen(url)
dataset = loadtxt(raw_data, delimiter=',')
print(dataset.shape)