# This script generates the figures used in the thesis.

import numpy as np
import pandas as pd
from env import ExpEnv
from stable_baselines3 import PPO
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.metrics import mean_squared_error, silhouette_score, normalized_mutual_info_score
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")
import plotly.express as px
import plotly.io as pio
pio.kaleido.scope.mathjax = None

methods = ['Mean', 'Median', 'Most Frequent', 'K-Nearest Neighbours', 'RL']

def sil(x, nc):
    kmeans = KMeans(n_clusters=nc)
    kmeans.fit(x)
    return(silhouette_score(x, kmeans.labels_))

def nmi(x, y, nc):
    kmeans = KMeans(n_clusters=nc)
    kmeans.fit(x)
    return(normalized_mutual_info_score(y, kmeans.labels_))

def calculate_metrics(n_samples, n_features, n_centers, missing_ratio, loaded_model):

    rmses, sils, nmis = {}, {}, {}

    mean_rmse, med_rmse, mf_rmse, knn_rmse, rl_rmse = [], [], [], [], []
    mean_sil, med_sil, mf_sil, knn_sil, rl_sil, gt_sil = [], [], [], [], [], []
    mean_nmi, med_nmi, mf_nmi, knn_nmi, rl_nmi, gt_nmi = [], [], [], [], [], []

    env = ExpEnv(n_samples=n_samples, n_features=n_features, n_centers=n_centers, missing_ratio=missing_ratio)

    for i in range(500):
        obs = env.reset()
        ground_truth = env.ground_truth
        X_miss = env.X_miss
        y = env.y

        X_mean = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(X_miss)
        X_med = SimpleImputer(missing_values=np.nan, strategy='median').fit_transform(X_miss)
        X_mf = SimpleImputer(missing_values=np.nan, strategy='most_frequent').fit_transform(X_miss)
        X_knn = KNNImputer(n_neighbors=5, weights="uniform").fit_transform(X_miss)

        gt_sil.append(sil(ground_truth, n_centers))
        gt_nmi.append(nmi(ground_truth, y, n_centers))

        mean_rmse.append(np.sqrt(mean_squared_error(ground_truth, X_mean)))
        mean_sil.append(sil(X_mean, n_centers))
        mean_nmi.append(nmi(X_mean, y, n_centers))

        med_rmse.append(np.sqrt(mean_squared_error(ground_truth, X_med)))
        med_sil.append(sil(X_med, n_centers))
        med_nmi.append(nmi(X_med, y, n_centers))

        mf_rmse.append(np.sqrt(mean_squared_error(ground_truth, X_mf)))
        mf_sil.append(sil(X_mf, n_centers))
        mf_nmi.append(nmi(X_mf, y, n_centers))

        knn_rmse.append(np.sqrt(mean_squared_error(ground_truth, X_knn)))
        knn_sil.append(sil(X_knn, n_centers))
        knn_nmi.append(nmi(X_knn, y, n_centers))

        done = False
        while not done:
            action, _ = loaded_model.predict(obs)
            obs, reward, done, _ = env.step(action)
            rl_rmse.append(np.sqrt(mean_squared_error(ground_truth, obs)))
            rl_sil.append(sil(obs, n_centers))
            rl_nmi.append(nmi(obs, y, n_centers))
        env.close()
    rmses['Mean'], rmses['Median'], rmses['Most Frequent'], rmses['K-Nearest Neighbours'], rmses['RL'] = mean_rmse, med_rmse, mf_rmse, knn_rmse, rl_rmse
    sils['Mean'], sils['Median'], sils['Most Frequent'], sils['K-Nearest Neighbours'], sils['RL'], sils['Ground Truth'] = mean_sil, med_sil, mf_sil, knn_sil, rl_sil, gt_sil
    nmis['Mean'], nmis['Median'], nmis['Most Frequent'], nmis['K-Nearest Neighbours'], nmis['RL'], nmis['Ground Truth'] = mean_nmi, med_nmi, mf_nmi, knn_nmi, rl_nmi, gt_nmi
    return(rmses, sils, nmis)

def train_and_calculate_metrics(n_samples, n_features, n_centers, missing_ratio, train):

    env = ExpEnv(n_samples=n_samples, n_features=n_features, n_centers=n_centers, missing_ratio=missing_ratio)

    if train:
        # Training model
        model = PPO("MlpPolicy", env, verbose=2)
        model.learn(total_timesteps=10000)
        model.save("trained_agent")

    # Evaluation
    loaded_model = PPO.load("trained_agent")
    rmses, sils, nmis = calculate_metrics(n_samples=n_samples, n_features=n_features, n_centers=n_centers, missing_ratio=missing_ratio, loaded_model=loaded_model)
    return(pd.DataFrame(rmses), pd.DataFrame(sils), pd.DataFrame(nmis))

#############
### PLOTS ###
#############

def rl_plot(n_samples, n_features, n_centers, missing_ratio, train):

    ### Reward Function
    def reward_function(rmse, rmse_mean):
        if rmse <= rmse_mean:
            return(np.cos(np.pi*rmse/(2*rmse_mean)))
        else:
            return(-1 + np.exp(-(rmse - rmse_mean)))
    rmse_mean = 1.5
    
    def f(rmse):
        return(reward_function(rmse,rmse_mean))
    X = np.linspace(0, 5, 1001)
    fX = [f(x) for x in X]
    fig = px.line(x=X, y=fX, title="Reward function", template='none')
    fig.update_layout(
        xaxis_title = 'RMSE',
        yaxis_title="Reward",
    )
    fig.add_shape(type='line',
                x0=rmse_mean, y0=-1,
                x1=rmse_mean, y1=1,
                line=dict(color='red', width=2))
    fig.add_annotation(text='RMSE with ground truth of the mean imputation',
                    x=rmse_mean, y=1,
                    xanchor='left', yanchor='bottom',
                    showarrow=False,
                    font=dict(color='red'))
    pio.write_image(fig, 'figures/reward_function.pdf')

    rmses, sils, nmis = train_and_calculate_metrics(n_samples, n_features, n_centers, missing_ratio, train)

    ### RMSES
    fig = px.box(rmses)
    fig.update_layout(
        title='RMSE with ground truth by Method',
        xaxis_title='Method',
        yaxis_title='RMSE with ground truth',
        template='none'
        )
    pio.write_image(fig, 'figures/rmse_by_method.pdf')

    ### SILS
    fig = px.box(sils)
    fig.update_layout(
        title=f'Silhouette Score by Method',
        xaxis_title='Method',
        yaxis_title='Silhouette Score',
        template='none'
        )
    pio.write_image(fig, 'figures/sil_by_method.pdf')

    ### NMIS
    fig = px.box(nmis)
    fig.update_layout(
        title='Normalized Mutual Info Score by Method',
        xaxis_title='Method',
        yaxis_title='Normalized Mutual Info Score',
        template='none'
        )
    pio.write_image(fig, 'figures/nmi_by_method.pdf')

rl_plot(n_samples=50, n_features=2, n_centers=2, missing_ratio=.01, train=True)