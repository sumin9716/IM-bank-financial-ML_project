import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

def run_clustering(panel: pd.DataFrame, feature_cols, k_kmeans=4, k_gmm=4, random_state=42):
    df = panel.copy()
    X = df[feature_cols].copy()
    for c in feature_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)
    X = (X - X.mean()) / (X.std(ddof=0) + 1e-12)
    km = KMeans(n_clusters=k_kmeans, n_init=10, random_state=random_state)
    df["cluster_kmeans"] = km.fit_predict(X.values)
    gmm = GaussianMixture(n_components=k_gmm, random_state=random_state)
    df["cluster_gmm"] = gmm.fit_predict(X.values)
    km_summary = df.groupby("cluster_kmeans")[feature_cols].mean().reset_index()
    gmm_summary = df.groupby("cluster_gmm")[feature_cols].mean().reset_index()
    return df, km_summary, gmm_summary
