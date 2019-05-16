'''
算法原理就是把多标记问题分解为单个分类问题
'''
from sklearn import datasets
from skmultilearn.problem_transform import BinaryRelevance
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
for i in range(4):
	for j in range(i + 1, 5):
		index = np.where(Y[:, i] != Y[:, j])
		Xt = X[index, :]
		Yt = Y[index, :]
		yt = Yt[0][:, i] > Yt[0][:, j]
		yt = yt + 0
		yt = np.transpose(np.matrix(yt))
		log = LogisticRegression()
		log.fit(Xt[0], yt)
		logs.append(log)

preds = np.zeros(Y.shape, dtype=np.int64)
k = 0
for i in range(4):
	for j in range(i + 1, 5):
		pred = logs[k].predict(X)
		pred = pred * 2 - 1

		index1 = np.transpose(np.matrix(np.where(pred > 0)))
		index2 = np.transpose(np.matrix(np.where(pred < 0)))
		preds[index1, i] = preds[index1, i] + np.matrix(pred[index1])
		preds[index2, i] = preds[index2, i] + np.matrix(pred[index2])
		preds[index1, j] = preds[index1, j] - np.matrix(pred[index1])
		preds[index2, j] = preds[index2, j] - np.matrix(pred[index2])
		k = k + 1
print(preds)
print(Y)
index = np.where(preds > 0)
preds[index[0], index[1]] = 1
index = np.where(preds < 0)
preds[index[0], index[1]] = 0
# print(preds)
# print(Y)
print(accuracy_score(preds, Y))
