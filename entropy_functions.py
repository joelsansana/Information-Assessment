"""
Per-feature information content via the log of the determinant of the
divergence between the empirical kernel-density estimate of each
variable and the uniform distribution on its support.

The script loads ``datasets/dataset3/csv_measurements.csv``, drops the
last 12 hours (720 samples at the 1-minute down-sampling stride used
in the project) and plots the LDDP score for every measurement
column except the time stamp and the lab-biodiesel concentration.
"""

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import entropy
from scipy.stats import gaussian_kde


def LDDP(data):
    """
    Compute the log of the determinant of the divergence between an
    empirical KDE and the uniform distribution on the same support.

    The score is

        ``log(n) - H(p_kernel, p_uniform)``

    where ``H`` is the Shannon cross-entropy (computed via
    :func:`scipy.stats.entropy`) between the KDE of ``data`` and the
    uniform distribution evaluated on ``n`` equally spaced points in
    ``[min(data), max(data)]``.

    Parameters
    ----------
    data : array-like of shape (n_samples,)
        Univariate samples.

    Returns
    -------
    score : float
        Log-determinant divergence of the empirical distribution with
        respect to the uniform reference. Higher values indicate that
        the variable carries more structure than a uniform signal.
    """
    kde = gaussian_kde(data)
    x = np.linspace(np.min(data), np.max(data), len(data))
    p_distribution = kde.pdf(x)
    m_distribution = np.array([1 for i in x]) / len(data)
    return np.log(len(data)) - entropy(p_distribution, m_distribution)


if __name__ == "__main__":
    df = pd.read_csv('datasets/dataset3/csv_measurements.csv')
    df = df.iloc[:-1:60, :]
    X = df.iloc[:-720, 1:-1]
    y = df.iloc[:-720, -1]

    Hx = [LDDP(X[col]) for col in X.columns]

    fig, ax = plt.subplots()
    sns.barplot(x=X.columns, y=Hx)
    plt.show()
