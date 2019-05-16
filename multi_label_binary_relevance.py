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
#因为做练习嘛，就直接写5，即5个标签
#得到训练的5个分类器
for i in range(5):
	logR = LogisticRegression()
	logR.fit(X, Y[:, i])
	logs.append(logR)

#用分类器进行预测
result = []
for i in range(5):
	res = logs[i].predict(X)
	result.append(res)

#将预测的结果转换成与Y一样的格式
fres = []
for i in range(len(result[0])):
	a = [result[0][i], result[1][i], result[2][i], result[3][i], result[4][i]]
	fres.append(a)

#精确度
fres = np.matrix(fres)
print(accuracy_score(fres, Y))

#使用线程函数的结果，两者一样
cl = BinaryRelevance(LogisticRegression())
cl.fit(data[0], data[1])
pred = cl.predict(data[0])
acc = accuracy_score(pred, data[1])
print(acc)
