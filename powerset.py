'''
算法原理就是把多标记问题分解为单个分类问题
'''
from sklearn import datasets
from skmultilearn.problem_transform import LabelPowerset
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import numpy as np 
import pickle
# 使用sklearn生成多标记数据，默认生成100个数据，5个标签，20个特征
# data = datasets.make_multilabel_classification()


def transfer(a):
	res = 0;
	le = len(a)
	for i in range(le):
		res = res + (2 ** (le - i - 1)) * a[i]
	return res


def transfer1(a):
	res = [0, 0, 0, 0, 0];
	i = 5;
	while i > 0:
		i = i - 1
		t = int(np.mod(a, 2))
		a = int(np.floor(a / 2))
		res[i] = t
	return res


test = datasets.make_multilabel_classification()
data = pickle.load(open('datasets.pickle', 'rb'))
X = data[0]
Y = data[1]

logs = []
yt = []
for i in range(Y.shape[0]):
	yt.append(transfer(Y[i, :])) 
log = LogisticRegression()
log.fit(X, yt)
p = log.predict(X)
res = []
for i in range(len(p)):
	rt = transfer1(p[i])
	res.append(rt)
print(accuracy_score(np.matrix(res), Y))



lb = LabelPowerset(LogisticRegression())
lb.fit(X, Y)
pred = lb.predict(X)
print(accuracy_score(pred, Y))