# This script defines the RL environment presented in the thesis.

import numpy as np
from gym import Env
from gym.spaces import Box
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_blobs
import warnings
warnings.filterwarnings("ignore")

def generate_data(n_samples, n_features, n_centers, missing_ratio, random_state=None):

    ground_truth, y = make_blobs(n_samples=n_samples, centers=n_centers, n_features=n_features, random_state=random_state)
    X_miss = np.copy(ground_truth)

    n_missing = int(n_samples * n_features * missing_ratio)
    indices = np.random.choice(n_samples * n_features, size=n_missing, replace=False)
    X_miss.flat[indices] = np.nan

    np.savetxt('data/ground_truth.csv', ground_truth, delimiter=',')
    np.savetxt('data/X_miss.csv', X_miss, delimiter=',')
    np.savetxt('data/y.csv', y, delimiter=',')

def general_impute(X, action, missing_indices):
    action_mat = np.zeros(X.shape)
    for i in range(len(missing_indices[0])):
        action_mat[missing_indices[0][i]][missing_indices[1][i]] = action[i]
    return(X + action_mat)

def reward_function(rmse, rmse_mean):
    if rmse <= rmse_mean:
        return(np.cos(np.pi*rmse/(2*rmse_mean)))
    else:
        return(-1 + np.exp(-(rmse - rmse_mean)))

class ExpEnv(Env):
    def __init__(self, n_samples, n_features, n_centers, missing_ratio):

        self.n_samples = n_samples
        self.n_features = n_features
        self.n_centers = n_centers
        self.missing_ratio = missing_ratio
        
        generate_data(self.n_samples, self.n_features, self.n_centers, self.missing_ratio)

        self.ground_truth = np.loadtxt('data/ground_truth.csv', delimiter=',')
        self.X_miss = np.loadtxt('data/X_miss.csv', delimiter=',')
        self.y = np.loadtxt('data/y.csv', delimiter=',')
        self.missing_indices = np.where(np.isnan(self.X_miss))

        self.action_space = Box(low=-2*np.nanmax(abs(self.X_miss[:,0])), high=2*np.nanmax(abs(self.X_miss[:,0])), shape=(len(self.missing_indices[1]),))
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.X_miss.shape[0],self.X_miss.shape[1]))
        self.state = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(self.X_miss)
        
        
    def step(self, action):
        rmse_mean = np.sqrt(mean_squared_error(self.ground_truth, self.state))
        # Action
        self.state = general_impute(self.state, action, self.missing_indices)
        # Reward
        rmse = np.sqrt(mean_squared_error(self.ground_truth, self.state))
        reward = reward_function(rmse, rmse_mean)
        # Done
        done = True
        # Info
        info = {}
        
        return self.state, reward, done, info

    def render(self):
        pass
    
    def reset(self):
        generate_data(self.n_samples, self.n_features, self.n_centers, self.missing_ratio)
        self.ground_truth = np.loadtxt('data/ground_truth.csv', delimiter=',')
        self.X_miss = np.loadtxt('data/X_miss.csv', delimiter=',')
        self.y = np.loadtxt('data/y.csv', delimiter=',')
        self.missing_indices = np.where(np.isnan(self.X_miss))
        self.state = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(self.X_miss)
        return self.state