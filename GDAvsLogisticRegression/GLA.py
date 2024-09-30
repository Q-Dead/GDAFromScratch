import numpy as np

class GaussianDiscriminantAnalysis:

    def __init__(self):
        self.phi = None
        self.sigma = None
        self.Mu = None
        self.Mu = None
        self.num_class = None
    
    def fit(self, X, y):
        i, n = X.shape
        self.num_class = np.unique(y).shape[0]
        self.Mu = np.zeros((self.num_class, n))
        self.phi = 0
        self.sigma = np.zeros((n , n))

        for label in range(self.num_class):

            self.Mu[label] = self.mean(X[y == label])
        
        self.phi = len(y[y==1]) / len(y)
        self.sigma = self.compute_sigma(X, y)

    def gaussian_density(self, X , mu, sigma):
        i , n = X.shape
        prob_density = np.zeros(i)
        for items in range(i):
            normalization= (np.linalg.pinv(sigma) @ (np.array(X)[items] - mu).T ) @ (np.array(X)[items] - mu)
            loglikelihood= (-0.5) * (np.log(np.linalg.det(sigma)) + normalization  + (n * np.log(2 * np.pi)))
            prob_density[items]= np.exp(loglikelihood)

        return prob_density
      
    
    def predict(self, X):
        n, k = X.shape
        prob_density = np.zeros((self.num_class, n))
        for label in range(self.num_class):
            tempcon = self.gaussian_density(X, self.Mu[label], self.sigma)
            prob_density[label] = tempcon * self.phi

        return np.argmax(prob_density.T, axis=1)
    
    def mean(self, X):
        i, n = X.shape
        container_mean = np.zeros(n)
        X_con = np.array(X.copy())

        for item in range(n):
            container_mean[item] = np.sum(X_con.T[item]) / i
        
        return container_mean
    
    def compute_sigma(self, X, y):
        n, m = X.shape
        X_con = np.array(X.copy())
        y_con = np.array(y.copy())
        Sigma = np.zeros((m, m))

        for i in range(n):
            if y_con[i] == 0:
                x_i = X_con[i].reshape(-1, 1)
                Sigma += (x_i - self.Mu[0]) @ (x_i - self.Mu[0]).T
            else:
                x_i = X_con[i].reshape(-1, 1)
                Sigma += (x_i - self.Mu[1]) @ (x_i - self.Mu[1]).T

        Sigma = (1/n) * Sigma

        return Sigma
