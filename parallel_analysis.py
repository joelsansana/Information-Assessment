"""
Utility functions for dimensionality and feature-relevance assessment.

Provides:
    * ``ParallelAnalysis`` -- Horn's parallel analysis to decide how many
      principal components are statistically meaningful.
    * ``FeatureRelevance_Pearson`` -- Pearson correlation with a permutation
      p-value.
    * ``FeatureRelevance_MI`` -- Mutual information with a permutation
      p-value.

All permutation-based procedures rely on ``np.random`` and are therefore
stochastic; set ``np.random.seed`` before calling for reproducibility.
"""

import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
from sklearn.feature_selection import mutual_info_regression


def ParallelAnalysis(data, percentile_tresh=90, n_iter=1000, n_components=None):
    """
    Horn's parallel analysis for principal component retention.

    The eigenvalues of the data's covariance matrix are compared against
    the distribution of eigenvalues obtained from ``n_iter`` random
    permutations of the columns. A component is retained when its
    eigenvalue exceeds the ``percentile_tresh``-th percentile of the
    null distribution at the same component index.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_features)
        Standardised feature matrix. The caller is responsible for
        normalising the data before invoking this function.
    percentile_tresh : int, default=90
        Percentile of the null distribution used as the threshold.
    n_iter : int, default=1000
        Number of permutations used to build the null distribution.
    n_components : int or None, default=None
        If given, restricts the PCA on the original data to this many
        components. The null distribution is always built with all
        available components.

    Returns
    -------
    original_eigenvalues : ndarray of shape (n_features,)
        Eigenvalues of the input data, ordered as returned by
        ``sklearn.decomposition.PCA``.
    threshold_eigenvalues : ndarray of shape (n_features,)
        The ``percentile_tresh`` percentile of the null eigenvalue
        distribution at each component index.
    n_components_to_retain : int
        Number of components whose eigenvalue is above the corresponding
        threshold.
    """
    pca = PCA(n_components=n_components)
    pca.fit(data)
    original_eigenvalues = pca.explained_variance_

    W = np.zeros(data.shape)
    pa = np.zeros((n_iter, data.shape[1]))
    for i in range(n_iter):
        for j in range(data.shape[1]):
            W[:, j] = data[np.random.permutation(data.shape[0]), j]
        pca_pa = PCA().fit(W)
        pa[i, :] = pca_pa.explained_variance_
    threshold_eigenvalues = np.percentile(pa, percentile_tresh, axis=0)

    n_components_to_retain = 0
    for i in range(len(original_eigenvalues)):
        if original_eigenvalues[i] > threshold_eigenvalues[i]:
            n_components_to_retain += 1

    return original_eigenvalues, threshold_eigenvalues, n_components_to_retain


def FeatureRelevance_Pearson(x, y, n_iter=100):
    """
    Pearson correlation between ``x`` and ``y`` with a permutation p-value.

    The p-value is computed as the fraction of randomly permuted
    ``y``-vectors whose absolute correlation with ``x`` is at least as
    large as the observed one. Two-sided testing is approximated by
    comparing against the same-signed tail of the null distribution.

    Parameters
    ----------
    x, y : array-like of shape (n_samples,)
        Paired samples.
    n_iter : int, default=100
        Number of permutations used to estimate the null distribution.

    Returns
    -------
    coefficient : float
        Pearson correlation coefficient between ``x`` and ``y``.
    pvalue : float
        Permutation-based p-value.
    """
    coefficient = pearsonr(x, y)[0]

    random_coefficient = [
        pearsonr(x, np.random.permutation(y))[0] for i in range(n_iter)
        ]
    if coefficient >= 0:
        pvalue = np.sum(random_coefficient >= coefficient) / n_iter
    else:
        pvalue = np.sum(random_coefficient <= coefficient) / n_iter
    return coefficient, pvalue


def FeatureRelevance_MI(x, y, n_iter=100):
    """
    Mutual information between ``x`` and ``y`` with a permutation p-value.

    Wraps :func:`sklearn.feature_selection.mutual_info_regression` and
    estimates significance by comparing the observed value to the
    distribution obtained from ``n_iter`` random permutations of ``y``.

    Parameters
    ----------
    x : array-like of shape (n_samples,)
        Predictor samples.
    y : array-like of shape (n_samples,)
        Target samples.
    n_iter : int, default=100
        Number of permutations used to estimate the null distribution.

    Returns
    -------
    coefficient : ndarray of shape (1,)
        Estimated mutual information between ``x`` and ``y``.
    pvalue : float
        Permutation-based p-value (fraction of null values >= observed).
    """
    coefficient = mutual_info_regression(x.reshape(-1, 1), y).astype('float64')

    random_coefficient = [
        mutual_info_regression(x.reshape(-1, 1), np.random.permutation(y))
        for i in range(n_iter)
        ]
    pvalue = np.sum(random_coefficient >= coefficient) / n_iter
    return coefficient, pvalue