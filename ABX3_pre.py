import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance
import time

pattens_data = ['element_A', 'element_B', 'element_X', 'melting_A', 'boiling_A',
                'electronegativity_A', 'ionization_A', 'melting_B', 'boiling_B', 'electronegativity_B',
                'ionization_B',
                'Density_A', 'at.wt._A', 'BCCenergy_pa_A', 'BCCfermi_A', 'BCCvolume_pa_A', 'GSenergy_pa_A',
                'ICSDVolume_A',
                'covalentradius_A', 'IonizationEnergy(kJ / mol)_A', 'AtomicRadius(Å)_A',
                'ElectronAffinity(kJ / mol)_A',
                'At.Radius(angstroms)_A', 'AtomicVolume(cm³ / mol)_A', 'FirstIonizationPotential(V)_A',
                'SecondIonizationPotential(V)_A', 'ThirdIonizationPotential(V)_A',
                'CoeficientofThermalExpansion(10 ^ -6K - 1)_A', 'specificheatcapacity_A', 'thermalconductivity_A',
                'heatoffusion_A', 'HeatofVaporization_A', 'At.# _A', 'Density_B', 'at.wt._B', 'BCCenergy_pa_B',
                'BCCfermi_B', 'BCCvolume_pa_B', 'GSenergy_pa_B', 'ICSDVolume_B', 'covalentradius_B',
                'IonizationEnergy(kJ / mol)_B', 'AtomicRadius(Å)_B', 'ElectronAffinity(kJ / mol)_B',
                'At.Radius(angstroms)_B', 'AtomicVolume(cm³ / mol)_B', 'FirstIonizationPotential(V)_B',
                'SecondIonizationPotential(V)_B', 'ThirdIonizationPotential(V)_B',
                'CoeficientofThermalExpansion(10 ^ -6K - 1)_B', 'specificheatcapacity_B', 'thermalconductivity_B',
                'heatoffusion_B', 'HeatofVaporization_B', 'At.# _B', 'Density_X', 'at.wt._X', 'BCCenergy_pa_X',
                'BCCfermi_X', 'BCCvolume_pa_X', 'GSenergy_pa_X', 'ICSDVolume_X', 'covalentradius_X',
                'IonizationEnergy_X', 'AtomicRadius_X', 'ElectronAffinity_X', 'At.Radius(angstroms)_X',
                'AtomicVolume_X', 'FirstIonizationPotential_X', 'SecondIonizationPotential_X',
                'ThirdIonizationPotential_X', 'CoeficientofThermalExpansion_X', 'specificheatcapacity_X',
                'thermalconductivity_X', 'heatoffusion_X', 'HeatofVaporization_X', 'At.# _X',
                'melting_X', 'boiling_X', 'electronegativity_X', 'ionization_X', 'A_radius', 'B_radius', 'radius_X',
                'formation energy per atom']

pattens_predict = ['element_A', 'element_B', 'element_X', 'melting_A', 'boiling_A',
                   'electronegativity_A', 'ionization_A', 'melting_B', 'boiling_B', 'electronegativity_B',
                   'ionization_B',
                   'Density_A', 'at.wt._A', 'BCCenergy_pa_A', 'BCCfermi_A', 'BCCvolume_pa_A', 'GSenergy_pa_A',
                   'ICSDVolume_A',
                   'covalentradius_A', 'IonizationEnergy(kJ / mol)_A', 'AtomicRadius(Å)_A',
                   'ElectronAffinity(kJ / mol)_A',
                   'At.Radius(angstroms)_A', 'AtomicVolume(cm³ / mol)_A', 'FirstIonizationPotential(V)_A',
                   'SecondIonizationPotential(V)_A', 'ThirdIonizationPotential(V)_A',
                   'CoeficientofThermalExpansion(10 ^ -6K - 1)_A', 'specificheatcapacity_A', 'thermalconductivity_A',
                   'heatoffusion_A', 'HeatofVaporization_A', 'At.# _A', 'Density_B', 'at.wt._B', 'BCCenergy_pa_B',
                   'BCCfermi_B', 'BCCvolume_pa_B', 'GSenergy_pa_B', 'ICSDVolume_B', 'covalentradius_B',
                   'IonizationEnergy(kJ / mol)_B', 'AtomicRadius(Å)_B', 'ElectronAffinity(kJ / mol)_B',
                   'At.Radius(angstroms)_B', 'AtomicVolume(cm³ / mol)_B', 'FirstIonizationPotential(V)_B',
                   'SecondIonizationPotential(V)_B', 'ThirdIonizationPotential(V)_B',
                   'CoeficientofThermalExpansion(10 ^ -6K - 1)_B', 'specificheatcapacity_B', 'thermalconductivity_B',
                   'heatoffusion_B', 'HeatofVaporization_B', 'At.# _B', 'Density_X', 'at.wt._X', 'BCCenergy_pa_X',
                   'BCCfermi_X', 'BCCvolume_pa_X', 'GSenergy_pa_X', 'ICSDVolume_X', 'covalentradius_X',
                   'IonizationEnergy_X', 'AtomicRadius_X', 'ElectronAffinity_X', 'At.Radius(angstroms)_X',
                   'AtomicVolume_X', 'FirstIonizationPotential_X', 'SecondIonizationPotential_X',
                   'ThirdIonizationPotential_X', 'CoeficientofThermalExpansion_X', 'specificheatcapacity_X',
                   'thermalconductivity_X', 'heatoffusion_X', 'HeatofVaporization_X', 'At.# _X',
                   'melting_X', 'boiling_X', 'electronegativity_X', 'ionization_X', 'A_radius', 'B_radius', 'radius_X',
                   ]

path_predict = 'D:/FJW/new_project2/NEW_DATA/new_ABX3_predict.txt'
path_data = 'D:/FJW/new_project2/NEW_DATA/new_ABX3.txt'

predict_data = pd.read_csv(path_predict, names=pattens_predict)
data = pd.read_csv(path_data, names=pattens_data)

cols = data.shape[1]
X = data.iloc[:, 3: cols - 1]
y = data.iloc[:, cols - 1: cols]
X = np.array(X)
y = np.array(y).ravel()

X_predict = predict_data.iloc[:, 3:]
X_predict = np.array(X_predict)

# 开始预测！
Y_pre = np.zeros(X_predict.shape[0],)
params = {'n_estimators': 183,  # 弱分类器的个数
          'max_depth': 3,  # 弱分类器（CART回归树）的最大深度
          'min_samples_split': 5,  # 分裂内部节点所需的最小样本数
          'learning_rate': 0.05,  # 学习率
          'loss': 'squared_error'}  # 损失函数类型
for i in range(1, 20):
    clf = GradientBoostingRegressor(**params)
    clf.fit(X, y)
    Y = clf.predict(X_predict)
    Y_pre = Y + Y_pre
    formation_predict = T = pow(10, 3.1529 - 0.3859 * ((Y_pre / 20) * 96.35 + 390.8) / 248.7) - 273.15

predict_data.insert(loc=len(predict_data.columns), column='formation_energy', value=formation_predict)
predict_data.to_csv('D:/FJW/newproject3/predict/new_ABX3_result.csv', index=False, sep=',')
