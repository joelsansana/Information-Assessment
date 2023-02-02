import numpy as np
from sklearn.decomposition import PCA

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