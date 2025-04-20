# -*- coding: utf-8 -*-
# Created on Sat Apr 19 21:19:26 2025

# @author: Nongnuch

# app.py
import streamlit as st
import pickle
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA

# Load model
with open('kmeans_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Set the page config
st.set_page_config(page_title="K-Means Clustering App", layout="centered")

# Set title
st.title("🔍 K-Means Clustering Visualizer")

# Display cluster centers
st.subheader("🧪 Example Data for Visualization")
st.markdown("This demo uses example data (2D PCA projection) to illustrate clustering results.")

# Generate synthetic 2D data
X, _ = make_blobs(n_samples=300, centers=loaded_model.n_clusters, cluster_std=0.60, random_state=0)

# Predict using the loaded model
y_kmeans = loaded_model.predict(X)

# Apply PCA to reduce to 2D for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
centers_pca = pca.transform(loaded_model.cluster_centers_)

# Plotting
fig, ax = plt.subplots()
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y_kmeans, cmap='viridis', alpha=0.6)
ax.scatter(centers_pca[:, 0], centers_pca[:, 1], s=300, c='red', marker='X', label='Cluster Centers')
ax.set_title('k-Means Clustering (2D PCA Projection)')
ax.set_xlabel('PCA Component 1')
ax.set_ylabel('PCA Component 2')
ax.legend()
st.pyplot(fig)
