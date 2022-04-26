from inspect import Parameter
import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels


class KernelRegression():
    def __init__(self, sigma=0.1):
        self.sigma = 0.1
        self.X = None
        self.y = None
        self.sigma = sigma

    def kernel_function(self, x1, x2):
        # return normal pdf
        return 1/(self.sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * np.linalg.norm(x1 - x2)**2 / self.sigma**2)

    def distance_function(self, x1, x2):
        return np.exp(-0.5 * np.linalg.norm(x1 - x2)**2 / self.sigma**2)

    def add_data(self, X, y):
        if self.X is None:
            self.X = X
            self.y = y
        else:
            self.X = np.vstack((self.X, X))
            self.y = np.concatenate((self.y, y))
    
    def predict(self, X_pred):
        K = pairwise_kernels(self.X, X_pred, metric=self.kernel_function)
        numerator = self.y.T @ K
        denominator = np.ones(K.shape[0]).T @ K

        return numerator/denominator

    def epistemic_uncertainty_wrong(self, X_pred):
        """
        This is wrong as it measures the variance, not variance of the mean.
        """
        K = pairwise_kernels(self.X, X_pred, metric=self.kernel_function)
        y_new = self.y**2 + self.sigma**2
        numerator = y_new.T @ K
        denominator = np.ones(K.shape[0]).T @ K

        return (numerator / denominator - self.predict(X_pred)**2)**0.5

    def epistemic_uncertainty(self, X_pred):
        """
        This is correct as it measures the variance of the mean.
        """
        counts = np.sum(pairwise_kernels(self.X, X_pred, self.distance_function), axis=0)
        return 1/(counts + 1)

class ParameterEstimatorKernel():
    def __init__(self, sigma=0.1):
        self.regressor = KernelRegression(sigma)
        self.uncertainty_regressor = KernelRegression(sigma)

    def reestimate(self, X, y):
        self.regressor.add_data(X, y)
        y_pred = self.regressor.predict(X)
        y_uncertainty = (y - y_pred)**2
        self.uncertainty_regressor.add_data(X, y_uncertainty)

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