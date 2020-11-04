import numpy as np

class LinearRegression:
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weigths = None
        self.bias = None,
    
    def fit(self, X, y):
        #we need to initiliaze the bias and weigths parameters
        n_samples, n_features = X.shape
        self.weigths = np.zeros(n_features) #set the zeros for each coefficient of x
        self.bias = 0 

        #gradient descent
        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.weigths) + self.bias
            
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred-y)

            self.weigths -= self.lr*dw
            self.bias -= self.lr*db

    def predict(self, X):
        y_pred = np.dot(X, self.weigths) + self.bias
        return y_pred

