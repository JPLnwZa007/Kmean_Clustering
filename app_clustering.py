# -*- coding: utf-8 -*-
# Created on Sat Apr 19 21:19:26 2025
# @author: Nongnuch

import streamlit as st
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Set the page config
st.set_page_config(page_title="K-Means Clustering App", layout="wide")

# Title
st.title("🔍 K-Means Clustering App with Iris Dataset")

# Sidebar slider
k = st.sidebar.slider("Select number of clusters (k)", min_value=2, max_value=10, value=4)

# Load the Iris dataset
iris = load_iris()
X = iris.data
X_scaled = StandardScaler().fit_transform(X)

# Reduce to 2D for visualization using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Apply k-means
kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled)

# Plotting
fig, ax = plt.subplots(figsize=(8, 5))
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y_kmeans, cmap='Set1')
centers_pca = pca.transform(kmeans.cluster_centers_)
ax.scatter(centers_pca[:, 0], centers_pca[:, 1], s=200, c='black', marker='X', label='Centroids')
ax.set_title('Clusters (2D PCA Projection)')
ax.set_xlabel('PCA1')
ax.set_ylabel('PCA2')
ax.legend(*scatter.legend_elements(), title="Clusters")
st.pyplot(fig)
