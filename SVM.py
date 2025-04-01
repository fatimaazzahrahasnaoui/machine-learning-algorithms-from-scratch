import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from mpl_toolkits.mplot3d import Axes3D

def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def svm(X_train, y_train, X_test, C=1):
    m, n = X_train.shape
    alpha = np.zeros(m)
    b = 0
    for _ in range(1000):
        for i in range(m):
            if y_train[i] * (np.dot(X_train[i], X_train.T.dot(alpha)) + b) < 1:
                alpha[i] += C * (1 - y_train[i] * (np.dot(X_train[i], X_train.T.dot(alpha)) + b))
    w = X_train.T.dot(alpha)
    return w, b

X, y = make_classification(n_samples=100, n_features=3, n_informative=2, n_redundant=0, n_classes=2)
X_train, X_test = X[:80], X[80:]
y_train, y_test = y[:80], y[80:]

w, b = svm(X_train, y_train, X_test)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_train[:, 0], X_train[:, 1], X_train[:, 2], c=y_train)
ax.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=y_test, marker='^')
plt.show()
