import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels


class KernelRegression():
    def __init__(self, sigma=0.1, prior=1, prior_strength=0.001):
        self.sigma = 0.1
        self.X = None
        self.y = None
        self.sigma = sigma
        # bias biases the predictions towards 1
        self.prior = prior # prior value
        self.prior_strength = prior_strength # prior strength

    def kernel_function(self, x1, x2):
        # return normal pdf
        return 1/(self.sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * np.linalg.norm(x1 - x2)**2 / self.sigma**2)

    def distance_function(self, x1, x2):
        return np.exp(-0.5 * np.linalg.norm(x1 - x2)**2 / self.sigma**2)

    def add_data(self, X, y):
        y = np.reshape(y, (-1, 1))
        if self.X is None:
            self.X = X
            self.y = y
        else:
            self.X = np.vstack((self.X, X))
            self.y = np.vstack((self.y, y))

    def replace_data(self, X, y):
        y = np.reshape(y, (-1, 1))
        self.X = X
        self.y = y
    
    def predict(self, X_pred):
        if self.X is None and self.prior_strength > 0:
            return self.prior * np.ones((X_pred.shape[0],1))

        K = pairwise_kernels(self.X, X_pred, metric=self.kernel_function)
        # print(self.X.shape, X_pred.shape, self.y.shape)
        numerator = np.matmul(self.y.T, K) + self.prior*self.prior_strength
        denominator = np.matmul(np.ones(K.shape[0]).T, K) + self.prior_strength

        return (numerator/denominator).T

    def epistemic_uncertainty_wrong(self, X_pred):
        """
        This is wrong as it measures the variance, not variance of the mean.
        """
        K = pairwise_kernels(self.X, X_pred, metric=self.kernel_function)
        y_new = self.y**2 + self.sigma**2
        numerator = np.matmul(y_new.T, K)
        denominator = np.matmul(np.ones(K.shape[0]).T, K)

        return (np.divide(numerator, denominator) - self.predict(X_pred)**2)**0.5

    def epistemic_uncertainty(self, X_pred):
        """
        This is correct as it measures the variance of the mean.
        """
        if self.X is None:
            return 1/self.prior_strength * np.ones((X_pred.shape[0],1))

        counts = np.sum(pairwise_kernels(self.X, X_pred, self.distance_function), axis=0)
        return 1/(counts + self.prior_strength)

class ParameterEstimatorKernel():
    def __init__(self, sigma=0.1):
        self.regressor = KernelRegression(sigma)
        self.uncertainty_regressor = KernelRegression(sigma, prior=0)

    def reestimate(self, X, y):
        self.regressor.add_data(X, y)
        y_pred = self.regressor.predict(self.regressor.X)
        # print("Reestimating, ypred shape", y_pred.shape, self.regressor.y.shape)
        y_uncertainty = (self.regressor.y - y_pred)**2
        self.uncertainty_regressor.replace_data(self.regressor.X, y_uncertainty)

    def predict(self, X_pred):
        y_pred = self.regressor.predict(X_pred)
        epistemic_uncertainty = self.regressor.epistemic_uncertainty(X_pred)
        y_uncertainty = self.uncertainty_regressor.predict(X_pred)
        return y_pred, epistemic_uncertainty, y_uncertainty

if __name__ == "__main__":
    # unit test
    X = np.linspace(0, 1, 20).reshape(-1, 1)
    # X += np.random.normal(0, 0.1, X.shape)
    y = np.sin(10*X) + np.random.normal(0, 0.3, X.shape)
    y = y.squeeze()

    # plot the data
    import matplotlib.pyplot as plt
    plt.plot(X, y, 'ro')
    plt.show()

    # create the kernel regressor
    regressor = ParameterEstimatorKernel()

    # add data
    regressor.reestimate(X, y)

    # predict
    X_pred = X = np.linspace(0, 1, 30).reshape(-1, 1) #np.array([[0], [1], [2], [3], [4]])

    y_pred, epistemic_uncertainty, y_uncertainty = regressor.predict(X_pred)

    # plot the prediction
    plt.plot(X_pred, y_pred, 'b-')
    plt.plot(X_pred, y_pred + epistemic_uncertainty, 'g-')
    plt.plot(X_pred, y_pred - epistemic_uncertainty, 'g-')
    plt.plot(X_pred, y_pred + y_uncertainty, 'r-')
    plt.plot(X_pred, y_pred - y_uncertainty, 'r-')
    plt.show()