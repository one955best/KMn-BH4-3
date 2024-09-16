import pattens as pt
from sklearn.svm import SVR
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import os
import time
from sklearn.model_selection import train_test_split

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
C, G, M, N = [], [], [], []
for c in np.linspace(3, 10, 8):
    for g in np.linspace(10 ** -9, 10 ** -6, 21):
        i, m, n = 0, 0, 0
        while i < 50:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6)
            clf = SVR(C=c, gamma=g)
            clf.fit(X_train, y_train)
            m = m + clf.score(X_test, y_test)
            n = n + clf.score(X_train, y_train)
            i = i + 1
        C.append(c)
        G.append(g)
        M.append(m / 50)
        N.append(n / 50)
svr_fit = time.time() - t0
print(svr_fit)

df = pd.DataFrame({'C': C, 'gamma': G, 'R^2_test': M, 'R^2_train': N})
df.to_csv('D:/FJW/project_artical/R^2_A2BCX6_SVR.csv', index=False, sep=',')

log_gamma = -np.log10(G)


for i in range(0, 8):
    plt.plot(log_gamma[21 * i:21 * i + 21], M[21 * i:21 * i + 21], label='C=' + str(i))
plt.xlabel('-log(gamma)')
plt.ylabel('R^2')
plt.show()
