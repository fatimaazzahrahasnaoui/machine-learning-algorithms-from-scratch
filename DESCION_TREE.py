import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from mpl_toolkits.mplot3d import Axes3D

class DecisionTree:
    def fit(self, X, y):
        self.tree = self.build_tree(X, y)
    
    def build_tree(self, X, y):
        m, n = X.shape
        if len(np.unique(y)) == 1:
            return {'label': np.unique(y)[0]}
        
        best_gini, best_split = float('inf'), None
        for feature in range(n):
            values = np.unique(X[:, feature])
            for value in values:
                left_mask = X[:, feature] <= value
                right_mask = ~left_mask
                left_y, right_y = y[left_mask], y[right_mask]
                gini = self.gini_impurity(left_y, right_y)
                if gini < best_gini:
                    best_gini = gini
                    best_split = (feature, value)
        
        left_mask = X[:, best_split[0]] <= best_split[1]
        right_mask = ~left_mask
        left_tree = self.build_tree(X[left_mask], y[left_mask])
        right_tree = self.build_tree(X[right_mask], y[right_mask])
        
        return {'feature': best_split[0], 'value': best_split[1], 'left': left_tree, 'right': right_tree}
    
    def gini_impurity(self, left, right):
        left_size = len(left)
        right_size = len(right)
        total_size = left_size + right_size
        p_left = left_size / total_size
        p_right = right_size / total_size
        return p_left * self.gini(left) + p_right * self.gini(right)
    
    def gini(self, y):
        classes = np.unique(y)
        p = [np.sum(y == c) / len(y) for c in classes]
        return 1 - sum(p_i**2 for p_i in p)
    
    def predict(self, X):
        return np.array([self.traverse_tree(x, self.tree) for x in X])
    
    def traverse_tree(self, x, tree):
        if 'label' in tree:
            return tree['label']
        if x[tree['feature']] <= tree['value']:
            return self.traverse_tree(x, tree['left'])
        else:
            return self.traverse_tree(x, tree['right'])


X, y = make_classification(n_samples=100, n_features=3, n_informative=2, n_redundant=0, n_classes=2)
X_train, X_test = X[:80], X[80:]
y_train, y_test = y[:80], y[80:]

dt = DecisionTree()
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_train[:, 0], X_train[:, 1], X_train[:, 2], c=y_train)
ax.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=y_pred, marker='^')
plt.show()
