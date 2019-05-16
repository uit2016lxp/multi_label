from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split

def distance(d1, d2):
	d = d1 - d2
	d = np.sum(d ** 2, axis=1)
	d = d ** 0.5
	return d


def knn(train, test, k, y_num, y):
	dis = distance(train, test)
	dis = dis[:, np.newaxis]
	y = y[:, np.newaxis]
	dis_y = np.hstack((dis, y))
	dis_y_t = []
	for i in range(dis_y.shape[0]):
		dis_y_t.append(tuple(dis_y[i, :]))
	dis_y_t = np.array(dis_y_t, dtype=[('x', float), ('y', int)])
	dis_y_t = np.sort(dis_y_t, order=['x', 'y'])
	dis_y_t = dis_y_t[0:k]
	res = np.linspace(0, 0, y_num)
	dis = []
	for i in range(len(dis_y_t)):
		dis.append(list(dis_y_t[i]))
	dis = np.array(dis)
	dis_sum = np.sum(dis, axis=0)
	dis[:, 0] = 1 - dis[:, 0] / dis_sum[0]
	print(dis)
	for i in range(k):
		res[int(dis[i, 1])] = res[int(dis[i, 1])] + dis[i, 0]
	result = np.argmax(res)
	return result


iris = datasets.load_iris()
X = iris.data
Y = iris.target
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
res = knn(X_train, X_test[0, :], 5, 3, Y_train)
print(res)
