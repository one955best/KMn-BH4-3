from sklearn.kernel_ridge import KernelRidge
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from sklearn.model_selection import train_test_split
import pattens as pt


pattens_data = pt.pattens_data_A2BCX6
pattens_predict = pt.pattens_predict_A2BCX6

path = 'D:/FJW/project_artical/new_A2BCX6.txt'

data = pd.read_csv(path, names=pattens_data)

cols = data.shape[1]
X = data.iloc[:, 4: cols - 1]
y = data.iloc[:, cols - 1: cols]
X = np.array(X)
y = np.array(y).ravel()

t0 = time.time()
A, G, M, N = [], [], [], []
for alpha in np.linspace(0.05, 0.2, 16):
    for g in np.linspace(10 ** -8, 10 ** -7, 21):
        i, m, n = 0, 0, 0
        while i < 50:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6)
            clf = KernelRidge(alpha=alpha, kernel='rbf', gamma=g)
            clf.fit(X_train, y_train)
            m = m + clf.score(X_test, y_test)
            n = n + clf.score(X_train, y_train)
            i = i + 1
        A.append(alpha)
        G.append(g)
        M.append(m / 50)
        N.append(n / 50)
        print(alpha)
time_fit = time.time() - t0

df = pd.DataFrame({'A': A, 'gamma': G, 'R^2_test': M, 'R^2_train': N})
df.to_csv('D:/FJW/project_artical/R^2_A2BCX6_KRR.csv', index=False, sep=',')
print(time_fit)

log_gamma = -np.log10(G)



for i in range(0, 8):
    plt.plot(log_gamma[21 * i:21 * i + 21], M[21 * i:21 * i + 21], label='C=' + str(i))
plt.xlabel('-log(gamma)')
plt.ylabel('R^2')
plt.show()
