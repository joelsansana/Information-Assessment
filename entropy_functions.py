import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import entropy
from scipy.stats import gaussian_kde

df = pd.read_csv('datasets/dataset3/csv_measurements.csv')
df = df.iloc[:-1:60, :]
X = df.iloc[:-720, 1:-1]
y = df.iloc[:-720, -1]
# X = X.values
# y = y.values

def LDDP(data):
    kde = gaussian_kde(data)
    x = np.linspace(np.min(data), np.max(data), len(data))
    p_distribution = kde.pdf(x)
    m_distribution = np.array([1 for i in x])/len(data)
    return np.log(len(data)) - entropy(p_distribution, m_distribution)

Hx = [LDDP(X[col]) for col in X.columns]

fig, ax = plt.subplots()
sns.barplot(x=X.columns, y=Hx)
plt.show()
