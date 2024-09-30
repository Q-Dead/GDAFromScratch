import numpy as np

class LogisticRegression:

    def __init__(self, learning_rate=0.01, n_iter=1000):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.weight = None
        self.intercept = None

    def fit(self, X, y):
        i, n = X.shape
        self.weight = np.zeros(n)
        self.intercept = 0

        for _ in range(self.n_iter):

            z = self.thetaTX(X)
            yhat = self.sigmoid(z)

            dbda = (1/i) * np.dot(X.T, (yhat - y))
            dbdb = (1/i) * np.sum(yhat - y)

            self.weight = self.weight - self.learning_rate * dbda
            self.intercept = self.intercept - self.learning_rate * dbdb
            

    def predict(self, X):
        Z = self.thetaTX(X)
        H = self.sigmoid(Z)
        return [1 if i >= 0.5 else 0 for i in H] 

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def thetaTX(self, X):
        return self.intercept + np.dot(X, self.weight) 
        