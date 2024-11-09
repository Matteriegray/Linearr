import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

class logReg():

    def __init__(self, lr = 0.001, n_itr = 1000):
        self.lr = lr
        self.n_itr = n_itr
        self.weights = None
        self.bias = None

    def fit(self, X , y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_itr):
            y_pred = 1/(1+np.exp(-(np.dot(X, self.weights))+self.bias))

            dw = (1/n_samples) * np.dot(X.T , (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)

            self.weights = self.weights - self.lr * dw
            self.bias = self.bias + self.lr * db

    def pred(self, X):
        
        y_pred = 1/(1+np.exp(-(np.dot(X, self.weights))+self.bias))
        class_pred = [0 if y<=0.5 else 1 for y in y_pred]

        return class_pred
    

bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 34)

cls = logReg(lr = 0.065)

cls.fit(X_train, y_train)

y_pred = cls.pred(X_test)


def accuracy(y_pred, y_test):
    return np.sum(y_pred==y_test)/len(y_test)

acc = accuracy(y_pred, y_test)

print(acc)
