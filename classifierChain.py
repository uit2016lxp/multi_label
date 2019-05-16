'''
每次学习时考虑前一个预测结果，即考虑相关性
需要根据其相关性对标签集合中的元素进行排序y1,y2...yn
这次练习直接按原顺序
'''
from sklearn import datasets
from skmultilearn.problem_transform import ClassifierChain
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import numpy as np 
import pickle
# 使用sklearn生成多标记数据，默认生成100个数据，5个标签，20个特征
data = datasets.make_multilabel_classification(n_samples=100)
pickle.dump(data, open('datasets.pickle', 'wb'))
# data = pickle.load(open('datasets.pickle', 'rb'))
X = data[0]
Y = data[1]
logs = []
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


for i in range(5):
	log = LogisticRegression()
	log.fit(np.hstack((X, Y[:, 0:i])), Y[:, i])# 每次训练将前一次的预测结果附带上
	logs.append(log)

results = []
for i in range(5):
	res = logs[i].predict(np.hstack((X, Y[:, 0:i])))
	results.append(res)

fres = []
for i in range(len(results[0])):
	a = [results[0][i], results[1][i], results[2][i], results[3][i], results[4][i]]
	fres.append(a)

fres = np.matrix(fres)
print(accuracy_score(fres, Y))
test = datasets.make_multilabel_classification()
# 使用已写好的分类器链算法验证结果
cl = ClassifierChain(LogisticRegression())
cl.fit(data[0], data[1])
pred = cl.predict(test[0])
print(accuracy_score(pred, test[1]))