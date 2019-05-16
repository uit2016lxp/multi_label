from sklearn import datasets
from skmultilearn.adapt import MLkNN
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import numpy as np 
import pickle


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


def distance(d1, d2):
	d = d1 - d2
	d = np.sum(d ** 2, axis=1)
	d = d ** 0.5
	return d


def get_k_neighbors(train, test, k, y_num, y):
	dis = distance(train, test)
	dis = dis[:, np.newaxis]

	dis_y = np.hstack((dis, y))
	dis_y_t = []
	for i in range(dis_y.shape[0]):
		dis_y_t.append(tuple(dis_y[i, :]))
	dis_y_t = np.array(dis_y_t, dtype=[('x', float), ('y1', int), ('y2', int), ('y3', int), ('y4', int), ('y5', int)])
	dis_y_t = np.sort(dis_y_t, order=['x'])
	dis_y_t_k = dis_y_t[0:k]
	dis = []
	for i in range(k):
		dis.append(list(dis_y_t_k[i]))
	dis = np.array(dis, dtype=int)
	return dis


def mlknn(train, test, k, y_num, y):
	s = 1.0
	ph = np.sum(y, axis=0)
	ph = (s + ph) / (s * 2 + train.shape[0])
	ph_ = 1 - ph
	peh0 = np.zeros((y_num, k + 1))
	peh1 = np.zeros((y_num, k + 1))
	for i in range(y_num):
		c0 = np.zeros(k + 1)
		c1 = np.zeros(k + 1)
		for j in range(train.shape[0]):
			temp = 0
			neighbors = get_k_neighbors(train, train[j, :], k, y_num, y)
			for t in range(k):
				temp = temp + neighbors[t][i + 1]
			if y[j][i] == 1:
				c1[temp] = c1[temp] + 1
			else:
				c0[temp] = c0[temp] + 1
		for j in range(k + 1):
			peh0[i, j] = (s + c0[j]) / (s * (k + 1) + np.sum(c0))
			peh1[i, j] = (s + c1[j]) / (s * (k + 1) + np.sum(c1))

	predicts = []
	for i in range(test.shape[0]):
		neigs = get_k_neighbors(train, test[i, :], k, y_num, y)
		predict = []
		for j in range(y_num):
			temp = 0
			for t in range(neigs.shape[0]):
				temp = temp + neigs[t][j + 1]
			if ph[j] * peh1[j, temp] > ph_[j] * peh0[j, temp]:
				predict.append(1)
			else:
				predict.append(0)
		predicts.append(predict)
	predicts = np.array(predicts)
	return predicts


data = pickle.load(open('datasets.pickle', 'rb'))
#得到训练数据X，和标签类别Y
X = data[0]
Y = data[1]

predict = mlknn(X, X, 8, 5, Y)
print(predict)
print(accuary(predict, Y))

ml = MLkNN(k=8)
ml.fit(X, Y)
p = ml.predict(X)
print(accuary(p, Y))

kn = KNeighborsClassifier(n_neighbors=8)
kn.fit(X, Y)
pp = kn.predict(X)
print(accuary(p, Y))