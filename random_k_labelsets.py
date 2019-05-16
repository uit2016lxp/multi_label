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
data = pickle.load(open('datasets.pickle', 'rb'))
test = datasets.make_multilabel_classification()
#得到训练数据X，和标签类别Y
def accuary(y_pred, y_true):
	res = 0
	for i in range(y_pred.shape[0]):
			temp = (y_pred[i, :] == y_true[i, :]) + 0
			inter_cnt = np.sum(temp)
			temp = (temp - 1) * (-1)
			union_cnt = np.sum(temp) * 2 + inter_cnt
			res = res + inter_cnt / union_cnt
	res = res / y_pred.shape[0]

	return np.round(res, 2)


X = data[0]
Y = data[1]
k = 3
n = 2 * len(Y[0, :])
indices = [0, 1, 2, 3, 4]
lbs = []
index_store = []
recode_cnt = np.linspace(0, 0, 5, dtype=np.int)
for i in range(n):
	np.random.shuffle(indices)
	index = indices[0:3]
	recode_cnt[index] = recode_cnt[index] + 1
	index_store.append(index)
	yt = Y[:, index]
	lb = LabelPowerset(LogisticRegression())
	lb.fit(X, yt)
	lbs.append(lb)

result = np.zeros(Y.shape)
for i in range(n):
	res = lbs[i].predict(X)
	index = index_store[i]
	result[:, index] = result[:, index] + res
# assert(i in index_store != 0)
result = result / recode_cnt
pred = (result > 0.5) + 0
print(accuary(pred, Y))