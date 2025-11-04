# Implement PCA 
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def perform_pca(X, n_components):
    """
    Perform Principal Component Analysis (PCA) on the dataset X.

    Parameters:
    X (numpy.ndarray): The input data matrix of shape (n_samples, n_features).
    n_components (int): The number of principal components to compute.

    Returns:
    X_pca (numpy.ndarray): The transformed data matrix of shape (n_samples, n_components).
    pca (PCA object): The fitted PCA object containing components and explained variance.
    """
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    return X_pca, pca

def plot_explained_variance_ratio(pca, save_path=None):
    """
    Plot the explained variance ratio of each principal component.

    Parameters:
    pca (PCA object): The fitted PCA object.
    """
    plt.figure(figsize=(8, 5))
    plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_)
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance Ratio by Principal Component')
    #save the plot
    plt.savefig(save_path)
    plt.close()


    