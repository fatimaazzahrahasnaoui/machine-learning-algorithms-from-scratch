import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from mpl_toolkits.mplot3d import Axes3D

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def knn(X_train, y_train, X_test, k):
    y_pred = []
    for x_test in X_test:
        distances = [euclidean_distance(x_test, x_train) for x_train in X_train]
        k_indices = np.argsort(distances)[:k]
        k_nearest_labels = [y_train[i] for i in k_indices]
        most_common = np.bincount(k_nearest_labels).argmax()
        y_pred.append(most_common)
    return np.array(y_pred)

X, y = make_classification(n_samples=100, n_features=3, n_informative=2, n_redundant=0, n_classes=2)
X_train, X_test = X[:80], X[80:]
y_train, y_test = y[:80], y[80:]

y_pred = knn(X_train, y_train, X_test, k=3)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_train[:, 0], X_train[:, 1], X_train[:, 2], c=y_train)
ax.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=y_pred, marker='^')
plt.show()
