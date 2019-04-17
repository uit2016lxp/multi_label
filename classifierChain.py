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
# 使用sklearn生成多标记数据，默认生成100个数据，5个标签，20个特征
data = datasets.make_multilabel_classification()

X = data[0]
Y = data[1]
logs = []

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

# 使用已写好的分类器链算法验证结果
cl = ClassifierChain(LogisticRegression())
cl.fit(data[0], data[1])
pred = cl.predict(data[0])
print(accuracy_score(pred, Y))