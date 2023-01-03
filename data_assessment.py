#%% Init
import time
start = time.time()

import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt
from pylab import rcParams

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

# %%
end = time.time()
print('Run time: {:.0f} seconds'.format(end-start))
plt.show()
