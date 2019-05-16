from sklearn import datasets
from skmultilearn.adapt import 
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import numpy as np 
import pickle
# 使用sklearn生成多标记数据，默认生成100个数据，5个标签，20个特征
# data = datasets.make_multilabel_classification()
data = pickle.load(open('datasets.pickle', 'rb'))
#得到训练数据X，和标签类别Y
X = data[0]
Y = data[1]
logs = []