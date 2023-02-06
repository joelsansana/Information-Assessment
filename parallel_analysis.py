import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
from sklearn.feature_selection import mutual_info_regression

def ParallelAnalysis(data, percentile_tresh=90, n_iter=1000, n_components=None):
    # Perform PCA on the normalized data
    pca = PCA(n_components=n_components)
    pca.fit(data)
    original_eigenvalues = pca.explained_variance_
    
    # Compute the threshold eigenvalue
    W = np.zeros(data.shape)
    pa = np.zeros((n_iter, data.shape[1]))
    for i in range(n_iter):
        for j in range(data.shape[1]):
            W[:, j] = data[np.random.permutation(data.shape[0]), j]
        pca_pa = PCA().fit(W)
        pa[i, :] = pca_pa.explained_variance_
    threshold_eigenvalues = np.percentile(pa, percentile_tresh, axis=0)
    
    # Determine the number of components to retain
    n_components_to_retain = 0
    for i in range(len(original_eigenvalues)):
        if original_eigenvalues[i] > threshold_eigenvalues[i]:
            n_components_to_retain += 1
    
    return original_eigenvalues, threshold_eigenvalues, n_components_to_retain

def FeatureRelevance_Pearson(x, y, n_iter=100):
    coefficient = pearsonr(x, y)[0]
    
    random_coefficient = [
        pearsonr(x, np.random.permutation(y))[0] for i in range(n_iter)
        ]
    
    return np.sum(np.abs(random_coefficient) >= np.abs(coefficient))/n_iter

def FeatureRelevance_MI(x, y, n_iter=100):
    coefficient = mutual_info_regression(x.reshape(-1, 1), y)
    
    random_coefficient = [
        mutual_info_regression(x.reshape(-1, 1), np.random.permutation(y)) for i in range(n_iter)
        ]
    
    return np.sum(random_coefficient >= coefficient)/n_iter