import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tslearn.clustering import KShape
from tslearn.clustering import TimeSeriesKMeans
from tslearn.clustering import silhouette_score

def check_data(df): 
    all_nan = [col for col in df if df[col].isna().all()]
    empty = [col for col in df if df[col].empty]
    if all_nan or empty:
        raise ValueError("Data check failed. Please review the data.")

def plot_silhouette_scores(scores, k_range, algorithm):
    plt.figure(figsize=(10, 5))
    plt.plot(k_range, scores, marker='o')
    plt.title('Silhouette Scores for Different k Values')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.xticks(k_range)
    plt.grid()
    plt.savefig(f'{algorithm}_silhouette_scores.pdf')
    plt.close()

# import data
df = pd.read_csv('data/electricity.csv', sep=';')
df.drop(['Timestamp'], axis=1, inplace=True) 

for col in df.columns:
    if df[col].dtype == 'object' or pd.api.types.is_string_dtype(df[col]):
        # Rimozione caratteri indesiderati
        df[col] = df[col].astype(str)  # forza tutto a stringa
        df[col] = df[col].str.replace(',', '.', regex=False)
        df[col] = df[col].str.strip()
        df[col] = df[col].str.replace(r'[^\d\.\-eE+]', '', regex=True)

df = df.apply(pd.to_numeric, errors='coerce')
check_data(df)

X = df.values.T

# z-standardization (mean=0, std=1)
X_mean = X.mean(axis=1, keepdims=True)
X_std = X.std(axis=1, keepdims=True)
X_norm = (X - X_mean) / X_std

print(X_norm.shape) # stampa (370, 140256)

X_hourly = X_norm[:, ::12] # downsamopling to one sample every 3 hours

X_train = X_hourly[:, :, None]  # to 3D array for tslearn (n_series, timesteps, dim) 

# k-means with DTW distance
silhouette_scores = []
print("------------------- k-means with DTW distance -------------------")
for k in range(2, 10):
    labels_k = TimeSeriesKMeans(n_clusters=k, metric="dtw", max_iter=5).fit_predict(X_train)
    sil = silhouette_score(X_train, labels_k, metric="dtw")
    print(f'For k={k} --> silhouette score={sil:.3f}')
    silhouette_scores.append(sil)

plot_silhouette_scores(silhouette_score, list(range(2, 10)), 'k-means')

#k-Shape
silhouette_scores = []
print("------------------- k-Shape -------------------")
for k in range(2, 10):
    labels_k = KShape(n_clusters=4, n_init=2, random_state=0).fit_predict(X_train)
    sil = silhouette_score(X_train, labels_k, metric="dtw")
    print(f'For k={k} --> silhouette score={sil:.3f}')

plot_silhouette_scores(silhouette_score, list(range(2, 20)), 'k-shape')