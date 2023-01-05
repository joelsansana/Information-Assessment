#%% Init
import time
start = time.time()

import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt
from pylab import rcParams
from scipy.cluster import hierarchy as hc
from scipy.special import rel_entr
from scipy.stats import pearsonr
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler

plt.style.use('seaborn-whitegrid')
rcParams['figure.figsize'] = 15, 8

import numpy as np
from sklearn.decomposition import PCA

def parallel_analysis(data, n_iter=1000, n_components=None):
    # Generate random data with the same number of samples and features
    random_data = np.random.normal(size=(data.shape[0], data.shape[1]))
    
    # Perform PCA on the random data
    pca = PCA(n_components=n_components)
    pca.fit(random_data)
    random_eigenvalues = pca.explained_variance_
    
    # Perform PCA on the original data
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
    threshold_eigenvalues = np.percentile(pa, 95, axis=0)
    
    # Determine the number of components to retain
    n_components_to_retain = 0
    for i in range(len(original_eigenvalues)):
        if original_eigenvalues[i] > threshold_eigenvalues[i]:
            n_components_to_retain += 1
    
    return original_eigenvalues, threshold_eigenvalues, n_components_to_retain

#%% Get data
df = pd.read_csv('datasets/dataset1/csv_measurements.csv')
df = df.iloc[:-1:60, :]

t = df['t/s']
X = df.iloc[:, 1:-1]
y = df.iloc[:, -1]

ts1 = t[:-720]
Xs1 = X.iloc[:-720, :]
ys1 = y[:-720]

df = pd.read_csv('datasets/dataset2/csv_measurements.csv')
df = df.iloc[:-1:60, :]

t = df['t/s']
X = df.iloc[:, 1:-1]
y = df.iloc[:, -1]

ts2 = t[:-720]
Xs2 = X.iloc[:-720, :]
ys2 = y[:-720]

df = pd.read_csv('datasets/dataset3/csv_measurements.csv')
df = df.iloc[:-1:60, :]

t = df['t/s']
X = df.iloc[:, 1:-1]
y = df.iloc[:, -1]

ts3 = t[:-720]
Xs3 = X.iloc[:-720, :]
ys3 = y[:-720]

#%% Visualization of key variables
fig, ax = plt.subplots()
ax.plot(ts1/(3600*24), ys1, label='1 grade')
ax.plot(ts2/(3600*24), ys2, label='3 grades')
ax.plot(ts3/(3600*24), ys3, label='DoE')
ax.set_xlabel('Time / days', fontsize=18)
ax.set_ylabel('xRE / -', fontsize=18)
ax.set_title('Train set', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(loc='best')

fig, ax = plt.subplots()
ax.plot(ts1/(3600*24), Xs1['TR/C'], label='1 grade')
ax.plot(ts2/(3600*24), Xs2['TR/C'], label='3 grade')
ax.plot(ts3/(3600*24), Xs3['TR/C'], label='DoE')
ax.set_xlabel('Time / days', fontsize=18)
ax.set_ylabel('Reactor Temperature / oC', fontsize=18)
ax.set_title('Train set', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(loc='best')

fig, ax = plt.subplots()
ax.plot(ts1/(3600*24), Xs1['Foil/(kg/h)'], label='1 grade')
ax.plot(ts2/(3600*24), Xs2['Foil/(kg/h)'], label='3 grade')
ax.plot(ts3/(3600*24), Xs3['Foil/(kg/h)'], label='DoE')
ax.set_xlabel('Time / days', fontsize=18)
ax.set_ylabel('Oil feed flow / kg/h', fontsize=18)
ax.set_title('Train set', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(loc='best')

#%% Colinearity (of X)
Z1 = StandardScaler().fit_transform(Xs1)
Z2 = StandardScaler().fit_transform(Xs2)
Z3 = StandardScaler().fit_transform(Xs3)

eig1, pa1, pc1 = parallel_analysis(Z1, n_iter=100)
eig2, pa2, pc2 = parallel_analysis(Z2, n_iter=100)
eig3, pa3, pc3 = parallel_analysis(Z3, n_iter=100)

fig, ax = plt.subplots(nrows=1, ncols=3)
ax[0].plot(eig1, '-o', label='Eigenvalues')
ax[1].plot(eig2, '-o', label='Eigenvalues')
ax[2].plot(eig3, '-o', label='Eigenvalues')
ax[0].plot(pa1, '-o', label='PA threshold')
ax[1].plot(pa2, '-o', label='PA threshold')
ax[2].plot(pa3, '-o', label='PA threshold')
ax[0].set_title('1 grade', fontsize=18)
ax[1].set_title('3 grades', fontsize=18)
ax[2].set_title('DoE', fontsize=18)
ax[0].legend(loc='upper right', fontsize=16)
ax[1].legend(loc='upper right', fontsize=16)
ax[2].legend(loc='upper right', fontsize=16)

C1 = pd.DataFrame(
    data=np.corrcoef(Xs1, rowvar=False), 
    columns=Xs1.columns, 
    index=Xs1.columns
    )

C2 = pd.DataFrame(
    data=np.corrcoef(Xs2, rowvar=False), 
    columns=Xs2.columns, 
    index=Xs2.columns
    )

C3 = pd.DataFrame(
    data=np.corrcoef(Xs3, rowvar=False), 
    columns=Xs3.columns, 
    index=Xs3.columns
    )

fig, ax = plt.subplots(nrows=1, ncols=3)
sns.heatmap(C1, vmin=-1, vmax=1, annot=True, fmt=".2f", cmap="Spectral", ax=ax[0])
ax[0].set_title('1 grade', fontsize=18)
sns.heatmap(C2, vmin=-1, vmax=1, annot=True, fmt=".2f", cmap="Spectral", ax=ax[1])
ax[1].set_title('3 grades', fontsize=18)
sns.heatmap(C3, vmin=-1, vmax=1, annot=True, fmt=".2f", cmap="Spectral", ax=ax[2])
ax[2].set_title('DoE', fontsize=18)

#%% Sparsity of X effects with Y
c1 = [pearsonr(Xs1[col], ys1)[0] for col in Xs1.columns]
c2 = [pearsonr(Xs2[col], ys2)[0] for col in Xs2.columns]
c3 = [pearsonr(Xs3[col], ys3)[0] for col in Xs3.columns]

fig, ax = plt.subplots(nrows=3, ncols=1)
sns.barplot(x=Xs1.columns, y=c1, ax=ax[0])
sns.barplot(x=Xs2.columns, y=c2, ax=ax[1])
sns.barplot(x=Xs3.columns, y=c3, ax=ax[2])
ax[0].set_ylabel('Corr - 1 grade', fontsize=18)
ax[1].set_ylabel('Corr - 3 grades', fontsize=18)
ax[2].set_ylabel('Corr - DoE', fontsize=18)

#%% Nonlinearity
mi1 = mutual_info_regression(Xs1, ys1)
mi1 /= mi1.max()
mi2 = mutual_info_regression(Xs2, ys2)
mi2 /= mi2.max()
mi3 = mutual_info_regression(Xs3, ys3)
mi3 /= mi3.max()

fig, ax = plt.subplots(nrows=3, ncols=1)
sns.barplot(x=Xs1.columns, y=mi1, ax=ax[0])
sns.barplot(x=Xs2.columns, y=mi2, ax=ax[1])
sns.barplot(x=Xs3.columns, y=mi3, ax=ax[2])
ax[0].set_ylabel('MI - 1 grade', fontsize=18)
ax[1].set_ylabel('MI - 3 grades', fontsize=18)
ax[2].set_ylabel('MI - DoE', fontsize=18)

#%% HC
fig, ax = plt.subplots(nrows=1, ncols=3)
hc.dendrogram(
    hc.linkage(Z1, method='ward'), 
    truncate_mode='lastp',
    p=10,
    color_threshold=75, 
    above_threshold_color='y', 
    ax=ax[0]
    )
hc.dendrogram(
    hc.linkage(Z2, method='ward'), 
    truncate_mode='lastp',
    p=10,
    color_threshold=75, 
    above_threshold_color='y', 
    ax=ax[1]
    )
hc.dendrogram(
    hc.linkage(Z3, method='ward'), 
    truncate_mode='lastp',
    p=10,
    color_threshold=75, 
    above_threshold_color='y', 
    ax=ax[2]
    )
ax[0].set_title('1 grade', fontsize=18)
ax[1].set_title('3 grades', fontsize=18)
ax[2].set_title('DoE', fontsize=18)
plt.xticks(rotation=90)
plt.tight_layout()

# cluster = AgglomerativeClustering(
#     n_clusters=3, 
#     affinity='euclidean', 
#     linkage='ward'
#     )
# hc_class = cluster.fit_predict(Z)

# # Plot scores HC
# dfT['HC class'] = hc_class
# sns.pairplot(
#     data=dfT,
#     vars=[f'PC{i}' for i in range(1,2+1)],
#     hue='HC class',
#     palette='tab10'
#     )

#%% Entropy
# Fit a KDE to the data
kde1 = KernelDensity(kernel='gaussian', bandwidth=0.75).fit(ys1.values.reshape(-1, 1))
kde2 = KernelDensity(kernel='gaussian', bandwidth=0.75).fit(ys2.values.reshape(-1, 1))
kde3 = KernelDensity(kernel='gaussian', bandwidth=0.75).fit(ys3.values.reshape(-1, 1))

# # Use the KDE to estimate the probability distribution of the data
# x_min, x_max = data.min(), data.max()
# x = np.linspace(x_min, x_max, 1000)
# log_prob = kde.score_samples(x.reshape(-1, 1))
# prob = np.exp(log_prob)

kl1 = rel_entr(ys1.values, ys1.values).sum() # ou Z-score
kl2 = rel_entr(ys2.values, ys2.values).sum()
kl3 = rel_entr(ys3.values, ys3.values).sum()

fig, ax = plt.subplots()
sns.barplot(
    x=[ys1.name, ys2.name, ys3.name,], 
    y=[kl1, kl2, kl3,], 
    ax=ax
    )

end = time.time()
print('Run time: {:.0f} seconds'.format(end-start))
plt.show()
