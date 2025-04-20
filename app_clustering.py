# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 21:19:26 2025

@author: Nongnuch
"""

import streamlit as st
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

# Set Streamlit page config
st.set_page_config(page_title="K-Means Clustering App", layout="centered")

# Title and description
st.title("🔍 K-Means Clustering App with Iris Dataset by Jhomphon Pothong")
st.markdown("This interactive app performs **K-Means clustering** on the Iris dataset and visualizes the results using **2D PCA projection**.")

# Sidebar for user input
st.sidebar.header("Configure Clustering")
k = st.sidebar.slider("Select number of clusters (k)", min_value=2, max_value=10, value=4)

# Load Iris dataset
data = load_iris()
X = data.data
features = data.feature_names

# Apply PCA to reduce to 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Train KMeans with selected k
kmeans = KMeans(n_clusters=k, random_state=0)
y_kmeans = kmeans.fit_predict(X)

# PCA projection of cluster centers
centers_pca = pca.transform(kmeans.cluster_centers_)

# Plotting
fig, ax = plt.subplots()
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y_kmeans, cmap='rainbow', s=50)
ax.set_title("Clusters (2D PCA Projection)")
ax.set_xlabel("PCA1")
ax.set_ylabel("PCA2")
legend_labels = [f"Cluster {i}" for i in range(k)]
legend = ax.legend(handles=scatter.legend_elements()[0], labels=legend_labels, title="Clusters")

# Show plot
st.pyplot(fig)
