import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor

class GaussianProcessRegressionWrapper():
    def __init__(self):
        self.gp_d = GaussianProcessRegressor(alpha=0.1)
        self.gp_d_sigma = GaussianProcessRegressor(alpha=0.1)
        self.visual_features = None
        self.d_values = None

    def add_data(self, visual_features, d_values, k_values):
        if self.visual_features is None:
            self.visual_features = visual_features
            self.d_values = d_values
        else:
            self.visual_features = np.vstack((self.visual_features, visual_features))
            self.d_values = np.concatenate((self.d_values, d_values))

    def fit(self):
        self.gp_d.fit(self.visual_features, self.d_values)
        preds = self.query(self.visual_features)
        residuals = (self.d_values - preds)**2
        self.gp_d_sigma.fit(self.visual_features, residuals)

    def query(self, visual_features):
        d_prediction, d_std = self.gp_d.predict(visual_features, return_std=True)
        return (d_prediction, d_std)

    def aleatoric_uncertainty(self, visual_features):
        d_sigma_prediction, d_sigma_std =  self.gp_d_sigma.predict(visual_features, return_std=True)
        return (d_sigma_prediction, d_sigma_std)