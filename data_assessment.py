#%% Init
import time
start = time.time()

import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt
from parallel_analysis import *
from pylab import rcParams
from scipy.stats import t as ttest
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from whitebox import Reactor

plt.style.use('seaborn-whitegrid')
rcParams['figure.figsize'] = 15, 8
rcParams["savefig.dpi"] = 300

#%% Get data
df = pd.read_csv('datasets/dataset1/csv_measurements.csv')
df = df.iloc[:-1:60, :]

t = df['t/s']
X = df.iloc[:, 1:-1]
y = df.iloc[:, -1]

ts1 = t[:-720]
Xs1 = X.iloc[:-720, :]
ys1 = y[:-720]

us1 = np.array([Xs1.iloc[:, 1], Xs1.iloc[:, 4], Xs1.iloc[:, 5]+273.15, \
    Xs1.iloc[:, 3]+273.15])

df = pd.read_csv('datasets/dataset2/csv_measurements.csv')
df = df.iloc[:-1:60, :]

t = df['t/s']
X = df.iloc[:, 1:-1]
y = df.iloc[:, -1]

ts2 = t[:-720]
Xs2 = X.iloc[:-720, :]
ys2 = y[:-720]

us2 = np.array([Xs2.iloc[:, 1], Xs2.iloc[:, 4], Xs2.iloc[:, 5]+273.15, \
    Xs2.iloc[:, 3]+273.15])

df = pd.read_csv('datasets/dataset3/csv_measurements.csv')
df = df.iloc[:-1:60, :]

t = df['t/s']
X = df.iloc[:, 1:-1]
y = df.iloc[:, -1]

ts3 = t[:-720]
Xs3 = X.iloc[:-720, :]
ys3 = y[:-720]

us3 = np.array([Xs3.iloc[:, 1], Xs3.iloc[:, 4], Xs3.iloc[:, 5]+273.15, \
    Xs3.iloc[:, 3]+273.15])

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

plt.show()

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

#%% First principle model
wb_model = Reactor()

par1 = wb_model.train(ts1, us1, ys1.values)
svR1 = wb_model.predict(ts1, par1, us1)

alpha = 0.05
residuals1 = svR1[:, 3] - ys1
dof = len(ts1) - len(par1)
t_value = par1[0] / (np.std(residuals1) / np.sqrt(dof))
p_value = ttest.sf(np.abs(t_value), dof)

if p_value < alpha:
    print("The reaction rate 1 is statistically significant with a p-value of {:.3f}".format(p_value))
else:
    print("The reaction rate 1 is not statistically significant with a p-value of {:.3f}".format(p_value))

par2 = wb_model.train(ts2, us2, ys2.values)
svR2 = wb_model.predict(ts2, par2, us2)

residuals2 = svR2[:, 3] - ys2
dof = len(ts2) - len(par2)
t_value = par2[0] / (np.std(residuals2) / np.sqrt(dof))
p_value = ttest.sf(np.abs(t_value), dof)

if p_value < alpha:
    print("The reaction rate 2 is statistically significant with a p-value of {:.3f}".format(p_value))
else:
    print("The reaction rate 2 is not statistically significant with a p-value of {:.3f}".format(p_value))
    
par3 = wb_model.train(ts3, us3, ys3.values)
svR3 = wb_model.predict(ts3, par3, us3)

residuals3 = svR3[:, 3] - ys3
dof = len(ts3) - len(par3)
t_value = par3[0] / (np.std(residuals3) / np.sqrt(dof))
p_value = ttest.sf(np.abs(t_value), dof)

if p_value < alpha:
    print("The reaction rate 3 is statistically significant with a p-value of {:.3f}".format(p_value))
else:
    print("The reaction rate 3 is not statistically significant with a p-value of {:.3f}".format(p_value))

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

plt.show()

#%% D2 - Structure
# Sparsity of X effects with Y
c1 = np.array([FeatureRelevance_Pearson(Xs1[col], ys1) for col in Xs1.columns])
c2 = np.array([FeatureRelevance_Pearson(Xs2[col], ys2) for col in Xs2.columns])
c3 = np.array([FeatureRelevance_Pearson(Xs3[col], ys3) for col in Xs3.columns])

rcParams['text.usetex'] = True
fig, ax = plt.subplots(nrows=3, ncols=1)
sns.barplot(x=Xs1.columns, y=np.where(c1[:,1]>0.05, 0, c1[:,0]), ax=ax[0])
sns.barplot(x=Xs2.columns, y=np.where(c2[:,1]>0.05, 0, c2[:,0]), ax=ax[1])
sns.barplot(x=Xs3.columns, y=np.where(c3[:,1]>0.05, 0, c3[:,0]), ax=ax[2])
ax[0].set_ylabel(r'$\rho$ - 1 grade', fontsize=16)
ax[1].set_ylabel(r'$\rho$ - 3 grades', fontsize=16)
ax[2].set_ylabel(r'$\rho$ - DoE', fontsize=16)
ax[0].set_title('Pearson Correlation Analysis', fontsize=18)
rcParams['text.usetex'] = False

# Nonlinearity of X effects with Y
mi1 = np.array([FeatureRelevance_MI(Xs1[col].values, ys1) for col in Xs1.columns]).astype('float64')
mi2 = np.array([FeatureRelevance_MI(Xs2[col].values, ys2) for col in Xs2.columns]).astype('float64')
mi3 = np.array([FeatureRelevance_MI(Xs3[col].values, ys3) for col in Xs3.columns]).astype('float64')

fig, ax = plt.subplots(nrows=3, ncols=1)
sns.barplot(x=Xs1.columns, y=np.where(mi1[:,1]>0.05, 0, mi1[:,0]), ax=ax[0])
sns.barplot(x=Xs2.columns, y=np.where(mi2[:,1]>0.05, 0, mi2[:,0]), ax=ax[1])
sns.barplot(x=Xs3.columns, y=np.where(mi3[:,1]>0.05, 0, mi3[:,0]), ax=ax[2])
ax[0].set_ylabel('MI - 1 grade', fontsize=16)
ax[1].set_ylabel('MI - 3 grades', fontsize=16)
ax[2].set_ylabel('MI - DoE', fontsize=16)
ax[0].set_title('Mutual Information Analysis', fontsize=18)

plt.show()

#%% D6 - Generalizability
# Information content / Data diversity
ssd = []
dbi = []
chi = []
ss = []

for k in range(2,10):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(Z1)
    ssd.append(kmeans.inertia_)
    dbi.append(davies_bouldin_score(Z1, kmeans.labels_))
    chi.append(calinski_harabasz_score(Z1, kmeans.labels_))
    ss.append(silhouette_score(Z1, kmeans.labels_, sample_size=300))

fig, ax = plt.subplots(nrows=2, ncols=2)
ax[0,0].plot(range(2, 10), ssd, 'o-', color='tab:blue')
ax[0,0].set_xlabel('# Clusters', fontsize=16) 
ax[0,0].set_ylabel('Sum of squared distances/Inertia', fontsize=16)
ax[1,0].plot(range(2, 10), chi, 'v-', color='tab:orange')
ax[1,0].set_xlabel('# Clusters', fontsize=16) 
ax[1,0].set_ylabel('Calinski-Harabasz Index', fontsize=16)
ax[0,1].plot(range(2, 10), dbi, 'd-', color='tab:green')
ax[0,1].set_xlabel('# Clusters', fontsize=16) 
ax[0,1].set_ylabel('Davies-Bouldin Index', fontsize=16)
ax[1,1].plot(range(2, 10), ss, 'x-', color='tab:red')
ax[1,1].set_xlabel('# Clusters', fontsize=16) 
ax[1,1].set_ylabel('Silhouette Coefficient', fontsize=16)
ax[0,0].set_title('Dataset 1 - 1 grade', fontsize=18)

ssd = []
dbi = []
chi = []
ss = []

for k in range(2,10):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(Z2)
    ssd.append(kmeans.inertia_)
    dbi.append(davies_bouldin_score(Z2, kmeans.labels_))
    chi.append(calinski_harabasz_score(Z2, kmeans.labels_))
    ss.append(silhouette_score(Z2, kmeans.labels_, sample_size=300))

fig, ax = plt.subplots(nrows=2, ncols=2)
ax[0,0].plot(range(2, 10), ssd, 'o-', color='tab:blue')
ax[0,0].set_xlabel('# Clusters', fontsize=16) 
ax[0,0].set_ylabel('Sum of squared distances/Inertia', fontsize=16)
ax[1,0].plot(range(2, 10), chi, 'v-', color='tab:orange')
ax[1,0].set_xlabel('# Clusters', fontsize=16) 
ax[1,0].set_ylabel('Calinski-Harabasz Index', fontsize=16)
ax[0,1].plot(range(2, 10), dbi, 'd-', color='tab:green')
ax[0,1].set_xlabel('# Clusters', fontsize=16) 
ax[0,1].set_ylabel('Davies-Bouldin Index', fontsize=16)
ax[1,1].plot(range(2, 10), ss, 'x-', color='tab:red')
ax[1,1].set_xlabel('# Clusters', fontsize=16) 
ax[1,1].set_ylabel('Silhouette Coefficient', fontsize=16)
ax[0,0].set_title('Dataset 2 - 3 grades', fontsize=18)

ssd = []
dbi = []
chi = []
ss = []

for k in range(2,10):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(Z3)
    ssd.append(kmeans.inertia_)
    dbi.append(davies_bouldin_score(Z3, kmeans.labels_))
    chi.append(calinski_harabasz_score(Z3, kmeans.labels_))
    ss.append(silhouette_score(Z3, kmeans.labels_, sample_size=300))

fig, ax = plt.subplots(nrows=2, ncols=2)
ax[0,0].plot(range(2, 10), ssd, 'o-', color='tab:blue')
ax[0,0].set_xlabel('# Clusters', fontsize=16) 
ax[0,0].set_ylabel('Sum of squared distances/Inertia', fontsize=16)
ax[1,0].plot(range(2, 10), chi, 'v-', color='tab:orange')
ax[1,0].set_xlabel('# Clusters', fontsize=16) 
ax[1,0].set_ylabel('Calinski-Harabasz Index', fontsize=16)
ax[0,1].plot(range(2, 10), dbi, 'd-', color='tab:green')
ax[0,1].set_xlabel('# Clusters', fontsize=16) 
ax[0,1].set_ylabel('Davies-Bouldin Index', fontsize=16)
ax[1,1].plot(range(2, 10), ss, 'x-', color='tab:red')
ax[1,1].set_xlabel('# Clusters', fontsize=16) 
ax[1,1].set_ylabel('Silhouette Coefficient', fontsize=16)
ax[0,0].set_title('Dataset 3 - DoE', fontsize=18)

end = time.time()
print('Run time: {:.0f} seconds'.format(end-start))
plt.show()
