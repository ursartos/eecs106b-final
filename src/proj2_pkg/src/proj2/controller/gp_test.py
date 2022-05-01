# import numpy as np
# from sklearn.gaussian_process import GaussianProcessRegressor

# data = np.array([
#     [0, 0],
#     [1, 1],
#     [1, 2],
#     [2, 4],
#     [4, 2]
# ])
# X = data[:, 0:1]
# y = data[:, 1:2]

# gpr = GaussianProcessRegressor().fit(X, y)

import numpy as np

X = np.linspace(start=0, stop=10, num=1_000).reshape(-1, 1)
y = np.squeeze(X * np.sin(X))

import matplotlib.pyplot as plt

plt.plot(X, y, label=r"$f(x) = x \sin(x)$", linestyle="dotted")
plt.legend()
plt.xlabel("$x$")
plt.ylabel("$f(x)$")
_ = plt.title("True generative process")
plt.show()

rng = np.random.RandomState(3)
training_indices = rng.choice(np.arange(y.size), size=1000, replace=False)
X_train, y_train = X[training_indices], y[training_indices]

# X_train = np.vstack((X_train, X_train))
# y_train = np.concatenate((y_train, y_train + 1*X_train[:6].squeeze()))

print(X_train.shape)

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

kernel = 1 * RBF(length_scale=0.1, length_scale_bounds=(1e-2, 1e2))
gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9, alpha=0.75 ** 2, n_jobs = -1)
gaussian_process.fit(X_train, y_train)
gaussian_process.kernel_

print("Done fitting")
mean_prediction, std_prediction = gaussian_process.predict(X, return_std=True)

plt.plot(X, y, label=r"$f(x) = x \sin(x)$", linestyle="dotted")
plt.scatter(X_train, y_train, label="Observations")
plt.plot(X, mean_prediction, label="Mean prediction")
plt.fill_between(
    X.ravel(),
    mean_prediction - 1.96 * std_prediction,
    mean_prediction + 1.96 * std_prediction,
    alpha=0.5,
    label=r"95% confidence interval",
)
plt.legend()
plt.xlabel("$x$")
plt.ylabel("$f(x)$")
_ = plt.title("Gaussian process regression on noise-free dataset")
plt.show()
