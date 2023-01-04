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
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler

plt.style.use('seaborn-whitegrid')
rcParams['figure.figsize'] = 15, 8

#%% Get data 1
df = pd.read_csv('datasets/dataset1/csv_measurements.csv')
df = df.iloc[:-1:60, :]

t = df['t/s']
X = df.iloc[:, 1:-1]
y = df.iloc[:, -1]

ts1 = t[:-720]
Xs1 = X.iloc[:-720, :]
ys1 = y[:-720]

#%% Get data 2
df = pd.read_csv('datasets/dataset2/csv_measurements.csv')
df = df.iloc[:-1:60, :]

t = df['t/s']
X = df.iloc[:, 1:-1]
y = df.iloc[:, -1]

ts2 = t[:-720]
Xs2 = X.iloc[:-720, :]
ys2 = y[:-720]

#%% Get data 3
df = pd.read_csv('datasets/dataset3/csv_measurements.csv')
df = df.iloc[:-1:60, :]

t = df['t/s']
X = df.iloc[:, 1:-1]
y = df.iloc[:, -1]

ts3 = t[:-720]
Xs3 = X.iloc[:-720, :]
ys3 = y[:-720]

#%% Exploratory Data Analysis
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

C1 = pd.DataFrame(
    data=np.corrcoef(pd.concat((Xs1, ys1), axis=1), rowvar=False), 
    columns=[pd.concat((Xs1, ys1), axis=1).columns], 
    index=np.append(Xs1.columns, 'KPI')
    )

C2 = pd.DataFrame(
    data=np.corrcoef(pd.concat((Xs2, ys2), axis=1), rowvar=False), 
    columns=[pd.concat((Xs2, ys2), axis=1).columns], 
    index=np.append(Xs2.columns, 'KPI')
    )

C3 = pd.DataFrame(
    data=np.corrcoef(pd.concat((Xs3, ys3), axis=1), rowvar=False), 
    columns=[pd.concat((Xs3, ys3), axis=1).columns], 
    index=np.append(Xs3.columns, 'KPI')
    )

fig, ax = plt.subplots(nrows=1, ncols=3)
sns.heatmap(C1, vmin=-1, vmax=1, annot=True, fmt=".2f", cmap="Spectral", ax=ax[0])
ax[0].set_title('1 grade', fontsize=18)
sns.heatmap(C2, vmin=-1, vmax=1, annot=True, fmt=".2f", cmap="Spectral", ax=ax[1])
ax[1].set_title('3 grades', fontsize=18)
sns.heatmap(C3, vmin=-1, vmax=1, annot=True, fmt=".2f", cmap="Spectral", ax=ax[2])
ax[2].set_title('DoE', fontsize=18)

mi1 = mutual_info_regression(Xs1, ys1)
mi2 = mutual_info_regression(Xs2, ys2)
mi3 = mutual_info_regression(Xs3, ys3)

fig, ax = plt.subplots(nrows=3, ncols=1)
sns.barplot(x=Xs1.columns, y=mi1, ax=ax[0])
sns.barplot(x=Xs2.columns, y=mi2, ax=ax[1])
sns.barplot(x=Xs3.columns, y=mi3, ax=ax[2])
ax[0].set_ylabel('MI - 1 grade', fontsize=18)
ax[1].set_ylabel('MI - 3 grades', fontsize=18)
ax[2].set_ylabel('MI - DoE', fontsize=18)

#%% PCA
Z1 = StandardScaler().fit_transform(Xs1)
Z2 = StandardScaler().fit_transform(Xs2)
Z3 = StandardScaler().fit_transform(Xs3)

pca1 = PCA().fit(Z1)
pca2 = PCA().fit(Z2)
pca3 = PCA().fit(Z3)
T1 = pca1.transform(Z1)
T2 = pca2.transform(Z2)
T3 = pca3.transform(Z3)
eig1 = pca1.explained_variance_
eig2 = pca2.explained_variance_
eig3 = pca3.explained_variance_

fig, ax = plt.subplots()
ax.plot(eig1, '-o', label='1 grade')
ax.plot(eig2, '-o', label='3 grades')
ax.plot(eig3, '-o', label='DoE')
ax.axhline(y=1, xmin=0, xmax=5, c='k', ls='--')#%% PCA
Z1 = StandardScaler().fit_transform(Xs1)
Z2 = StandardScaler().fit_transform(Xs2)
Z3 = StandardScaler().fit_transform(Xs3)

pca1 = PCA().fit(Z1)
pca2 = PCA().fit(Z2)
pca3 = PCA().fit(Z3)
T1 = pca1.transform(Z1)
T2 = pca2.transform(Z2)
T3 = pca3.transform(Z3)
eig1 = pca1.explained_variance_
eig2 = pca2.explained_variance_
plt.legend(loc='best')

dfT1 = pd.DataFrame(T1[:, :2], columns=['PC1', 'PC2'])
dfT1['Label'] = ys1.values
dfT2 = pd.DataFrame(T2[:, :2], columns=['PC1', 'PC2'])
dfT2['Label'] = ys2.values
dfT3 = pd.DataFrame(T3[:, :2], columns=['PC1', 'PC2'])
dfT3['Label'] = ys3.values
fig, ax = plt.subplots(nrows=1, ncols=3)
sns.scatterplot(data=dfT1, x='PC1', y='PC2', hue='Label', palette="Spectral", ax=ax[0])
ax[0].set_title('1 grade', fontsize=18)
sns.scatterplot(data=dfT2, x='PC1', y='PC2', hue='Label', palette="Spectral", ax=ax[1])
ax[1].set_title('3 grades', fontsize=18)
sns.scatterplot(data=dfT3, x='PC1', y='PC2', hue='Label', palette="Spectral", ax=ax[2])
ax[2].set_title('DoE', fontsize=18)

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
