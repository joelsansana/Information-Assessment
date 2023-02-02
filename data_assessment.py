#%% Init
import time
start = time.time()

import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt
from parallel_analysis import *
from pylab import rcParams
from scipy.cluster import hierarchy as hc
from scipy.special import rel_entr
from scipy.stats import entropy, gaussian_kde, pearsonr
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler

plt.style.use('seaborn-whitegrid')
rcParams['figure.figsize'] = 15, 8

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

#%% D1 - Resolution
# Granularity
## Count the number of data points
data_points1 = Xs1.shape[0]
print("Number of data points 1: ", data_points1)
## Calculate the number of unique values for each column
unique1 = Xs1.nunique()
print("Unique values 1: \n", unique1)
## Analyze the level of detail
min_time = Xs1.index.min()
max_time = Xs1.index.max()
time_diff1 = max_time - min_time
print("The measurement period is: ", time_diff1, "min")
print("The frequency of measurement is: ", data_points1/time_diff1, "measurement/min")

data_points2 = Xs2.shape[0]
print("Number of data points 2: ", data_points2)
data_points3 = Xs3.shape[0]
print("Number of data points 3: ", data_points3)

unique2 = Xs2.nunique()
print("Unique values 1: \n", unique2)
unique3 = Xs3.nunique()
print("Unique values 1: \n", unique3)

min_time = Xs2.index.min()
max_time = Xs2.index.max()
time_diff2 = max_time - min_time
print("The measurement period is: ", time_diff2, "min")
print("The frequency of measurement is: ", data_points2/time_diff2, "measurement/min")
min_time = Xs3.index.min()
max_time = Xs3.index.max()
time_diff3 = max_time - min_time
print("The measurement period is: ", time_diff3, "min")
print("The frequency of measurement is: ", data_points3/time_diff3, "measurement/min")

#%% D2 - Structure
# Colinearity (of X)
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

Z1 = StandardScaler().fit_transform(Xs1)
Z2 = StandardScaler().fit_transform(Xs2)
Z3 = StandardScaler().fit_transform(Xs3)

eig1, pa1, pc1 = ParallelAnalysis(Z1, n_iter=100)
eig2, pa2, pc2 = ParallelAnalysis(Z2, n_iter=100)
eig3, pa3, pc3 = ParallelAnalysis(Z3, n_iter=100)

col1 = (1 - pc1/Z1.shape[1])*100
col2 = (1 - pc2/Z2.shape[1])*100
col3 = (1 - pc3/Z3.shape[1])*100

fig, ax = plt.subplots(nrows=1, ncols=3)
ax[0].plot(eig1, '-o', label='Eigenvalues')
ax[1].plot(eig2, '-o', label='Eigenvalues')
ax[2].plot(eig3, '-o', label='Eigenvalues')
ax[0].plot(pa1, '-o', label='PA threshold')
ax[1].plot(pa2, '-o', label='PA threshold')
ax[2].plot(pa3, '-o', label='PA threshold')
ax[0].set_title('Colinearity = {:.0f}%'.format(col1), fontsize=18)
ax[1].set_title('Colinearity = {:.0f}%'.format(col2), fontsize=18)
ax[2].set_title('Colinearity = {:.0f}%'.format(col3), fontsize=18)
ax[0].legend(loc='upper right', fontsize=16)
ax[1].legend(loc='upper right', fontsize=16)
ax[2].legend(loc='upper right', fontsize=16)

# Sparsity of X effects with Y
c1 = [FeatureRelevance_Pearson(Xs1[col], ys1) for col in Xs1.columns]
c2 = [FeatureRelevance_Pearson(Xs2[col], ys2) for col in Xs2.columns]
c3 = [FeatureRelevance_Pearson(Xs3[col], ys3) for col in Xs3.columns]

fig, ax = plt.subplots(nrows=3, ncols=1)
sns.barplot(x=Xs1.columns, y=c1, ax=ax[0])
sns.barplot(x=Xs2.columns, y=c2, ax=ax[1])
sns.barplot(x=Xs3.columns, y=c3, ax=ax[2])
ax[0].set_ylabel('Corr - 1 grade', fontsize=18)
ax[1].set_ylabel('Corr - 3 grades', fontsize=18)
ax[2].set_ylabel('Corr - DoE', fontsize=18)

# Nonlinearity of X effects with Y
su1 = [FeatureRelevance_MI(Xs1[col].values, ys1) for col in Xs1.columns]
su2 = [FeatureRelevance_MI(Xs2[col].values, ys2) for col in Xs2.columns]
su3 = [FeatureRelevance_MI(Xs3[col].values, ys3) for col in Xs3.columns]

fig, ax = plt.subplots(nrows=3, ncols=1)
sns.barplot(x=Xs1.columns, y=su1, ax=ax[0])
sns.barplot(x=Xs2.columns, y=su2, ax=ax[1])
sns.barplot(x=Xs3.columns, y=su3, ax=ax[2])
ax[0].set_ylabel('SU - 1 grade', fontsize=18)
ax[1].set_ylabel('SU - 3 grades', fontsize=18)
ax[2].set_ylabel('SU - DoE', fontsize=18)

#%% D6 - Generalizability
# Information content / Data diversity
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

# #%% Entropy
# # Fit a KDE to the data
# kde1 = KernelDensity(kernel='gaussian', bandwidth=0.75).fit(ys1.values.reshape(-1, 1))
# kde2 = KernelDensity(kernel='gaussian', bandwidth=0.75).fit(ys2.values.reshape(-1, 1))
# kde3 = KernelDensity(kernel='gaussian', bandwidth=0.75).fit(ys3.values.reshape(-1, 1))

# # Use the KDE to estimate the probability distribution of the data
# y_min1, y_max1 = ys1.min(), ys1.max()
# y_min2, y_max2 = ys2.min(), ys2.max()
# y_min3, y_max3 = ys3.min(), ys3.max()

# x1 = np.linspace(y_min1, y_max1, 1000)
# x2 = np.linspace(y_min2, y_max2, 1000)
# x3 = np.linspace(y_min3, y_max3, 1000)

# log_prob1 = kde1.score_samples(x1.reshape(-1, 1))
# prob1 = np.exp(log_prob1)
# log_prob2 = kde2.score_samples(x2.reshape(-1, 1))
# prob2 = np.exp(log_prob2)
# log_prob3 = kde3.score_samples(x3.reshape(-1, 1))
# prob3 = np.exp(log_prob3)

# fig, ax = plt.subplots()
# ax.plot(x1, prob1)
# ax.plot(x2, prob2)
# ax.plot(x3, prob3)

# kl1 = rel_entr(ys1.values, ys1.values).sum() # ou Z-score
# kl2 = rel_entr(ys2.values, ys2.values).sum()
# kl3 = rel_entr(ys3.values, ys3.values).sum()

end = time.time()
print('Run time: {:.0f} seconds'.format(end-start))
plt.show()
