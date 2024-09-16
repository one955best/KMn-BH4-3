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

pattens_predict = ['A', 'B1', 'B2', 'X',  'melting_A', 'boiling_A',
                   'electronegativity_A', 'ionization_A', 'melting_B1', 'boiling_B1', 'electronegativity_B1',
                   'ionization_B1', 'melting_B2', 'boiling_B2', 'electronegativity_B2',
                   'ionization_B2',
                   'Density_A', 'at.wt._A', 'BCCenergy_pa_A', 'BCCfermi_A', 'BCCvolume_pa_A', 'GSenergy_pa_A',
                   'ICSDVolume_A',
                   'covalentradius_A', 'IonizationEnergy(kJ / mol)_A', 'AtomicRadius(Å)_A',
                   'ElectronAffinity(kJ / mol)_A',
                   'At.Radius(angstroms)_A', 'AtomicVolume(cm³ / mol)_A', 'FirstIonizationPotential(V)_A',
                   'SecondIonizationPotential(V)_A', 'ThirdIonizationPotential(V)_A',
                   'CoeficientofThermalExpansion(10 ^ -6K - 1)_A', 'specificheatcapacity_A', 'thermalconductivity_A',
                   'heatoffusion_A', 'HeatofVaporization_A', 'At.# _A', 'Density_B1', 'at.wt._B1', 'BCCenergy_pa_1B',
                   'BCCfermi_B1', 'BCCvolume_pa_B1', 'GSenergy_pa_B1', 'ICSDVolume_B1', 'covalentradius_B1',
                   'IonizationEnergy(kJ / mol)_B1', 'AtomicRadius(Å)_B1', 'ElectronAffinity(kJ / mol)_B1',
                   'At.Radius(angstroms)_B1', 'AtomicVolume(cm³ / mol)_B1', 'FirstIonizationPotential(V)_B1',
                   'SecondIonizationPotential(V)_B1', 'ThirdIonizationPotential(V)_B1',
                   'CoeficientofThermalExpansion(10 ^ -6K - 1)_B1', 'specificheatcapacity_B1', 'thermalconductivity_B1',
                   'heatoffusion_B1', 'HeatofVaporization_B1', 'At.# _B1', 'Density_B2', 'at.wt._B2', 'BCCenergy_pa_B2',
                   'BCCfermi_B2', 'BCCvolume_pa_B2', 'GSenergy_pa_B2', 'ICSDVolume_B2', 'covalentradius_B2',
                   'IonizationEnergy(kJ / mol)_B2', 'AtomicRadius(Å)_B2', 'ElectronAffinity(kJ / mol)_B2',
                   'At.Radius(angstroms)_B2', 'AtomicVolume(cm³ / mol)_B2', 'FirstIonizationPotential(V)_B2',
                   'SecondIonizationPotential(V)_B2', 'ThirdIonizationPotential(V)_B2',
                   'CoeficientofThermalExpansion(10 ^ -6K - 1)_B2', 'specificheatcapacity_B2', 'thermalconductivity_B2',
                   'heatoffusion_B2', 'HeatofVaporization_B2', 'At.# _B2', 'Density_X', 'at.wt._X', 'BCCenergy_pa_X',
                   'BCCfermi_X', 'BCCvolume_pa_X', 'GSenergy_pa_X', 'ICSDVolume_X', 'covalentradius_X',
                   'IonizationEnergy_X', 'AtomicRadius_X', 'ElectronAffinity_X', 'At.Radius(angstroms)_X',
                   'AtomicVolume_X', 'FirstIonizationPotential_X', 'SecondIonizationPotential_X',
                   'ThirdIonizationPotential_X', 'CoeficientofThermalExpansion_X', 'specificheatcapacity_X',
                   'thermalconductivity_X', 'heatoffusion_X', 'HeatofVaporization_X', 'At.# _X',
                   'melting_X', 'boiling_X', 'electronegativity_X', 'ionization_X', 'A_radius', 'B1_radius',
                   'B2_radius', 'X_radius',]

pattens_data = ['A', 'B1', 'B2', 'X',  'melting_A', 'boiling_A',
                'electronegativity_A', 'ionization_A', 'melting_B1', 'boiling_B1', 'electronegativity_B1',
                'ionization_B1', 'melting_B2', 'boiling_B2', 'electronegativity_B2',
                'ionization_B2',
                'Density_A', 'at.wt._A', 'BCCenergy_pa_A', 'BCCfermi_A', 'BCCvolume_pa_A', 'GSenergy_pa_A',
                'ICSDVolume_A',
                'covalentradius_A', 'IonizationEnergy(kJ / mol)_A', 'AtomicRadius(Å)_A',
                'ElectronAffinity(kJ / mol)_A',
                'At.Radius(angstroms)_A', 'AtomicVolume(cm³ / mol)_A', 'FirstIonizationPotential(V)_A',
                'SecondIonizationPotential(V)_A', 'ThirdIonizationPotential(V)_A',
                'CoeficientofThermalExpansion(10 ^ -6K - 1)_A', 'specificheatcapacity_A', 'thermalconductivity_A',
                'heatoffusion_A', 'HeatofVaporization_A', 'At.# _A', 'Density_B1', 'at.wt._B1', 'BCCenergy_pa_1B',
                'BCCfermi_B1', 'BCCvolume_pa_B1', 'GSenergy_pa_B1', 'ICSDVolume_B1', 'covalentradius_B1',
                'IonizationEnergy(kJ / mol)_B1', 'AtomicRadius(Å)_B1', 'ElectronAffinity(kJ / mol)_B1',
                'At.Radius(angstroms)_B1', 'AtomicVolume(cm³ / mol)_B1', 'FirstIonizationPotential(V)_B1',
                'SecondIonizationPotential(V)_B1', 'ThirdIonizationPotential(V)_B1',
                'CoeficientofThermalExpansion(10 ^ -6K - 1)_B1', 'specificheatcapacity_B1', 'thermalconductivity_B1',
                'heatoffusion_B1', 'HeatofVaporization_B1', 'At.# _B1', 'Density_B2', 'at.wt._B2', 'BCCenergy_pa_B2',
                'BCCfermi_B2', 'BCCvolume_pa_B2', 'GSenergy_pa_B2', 'ICSDVolume_B2', 'covalentradius_B2',
                'IonizationEnergy(kJ / mol)_B2', 'AtomicRadius(Å)_B2', 'ElectronAffinity(kJ / mol)_B2',
                'At.Radius(angstroms)_B2', 'AtomicVolume(cm³ / mol)_B2', 'FirstIonizationPotential(V)_B2',
                'SecondIonizationPotential(V)_B2', 'ThirdIonizationPotential(V)_B2',
                'CoeficientofThermalExpansion(10 ^ -6K - 1)_B2', 'specificheatcapacity_B2', 'thermalconductivity_B2',
                'heatoffusion_B2', 'HeatofVaporization_B2', 'At.# _B2', 'Density_X', 'at.wt._X', 'BCCenergy_pa_X',
                'BCCfermi_X', 'BCCvolume_pa_X', 'GSenergy_pa_X', 'ICSDVolume_X', 'covalentradius_X',
                'IonizationEnergy_X', 'AtomicRadius_X', 'ElectronAffinity_X', 'At.Radius(angstroms)_X',
                'AtomicVolume_X', 'FirstIonizationPotential_X', 'SecondIonizationPotential_X',
                'ThirdIonizationPotential_X', 'CoeficientofThermalExpansion_X', 'specificheatcapacity_X',
                'thermalconductivity_X', 'heatoffusion_X', 'HeatofVaporization_X', 'At.# _X',
                'melting_X', 'boiling_X', 'electronegativity_X', 'ionization_X', 'A_radius', 'B1_radius', 'B2_radius',
                'X_radius', 'formation energy per atom']

path_predict = 'D:/FJW/new_project2/NEW_DATA/new_A2BCX6_predict.txt'
path_data = 'D:/FJW/new_project2/NEW_DATA/new_A2BCX6.txt'

predict_data = pd.read_csv(path_predict, names=pattens_predict)
data = pd.read_csv(path_data, names=pattens_data)

cols = data.shape[1]
X = data.iloc[:, 4: cols - 1]
y = data.iloc[:, cols - 1: cols]
X = np.array(X)
y = np.array(y).ravel()

X_predict = predict_data.iloc[:, 4:]
X_predict = np.array(X_predict)



# 开始预测！

Y_pre = np.zeros(X_predict.shape[0],)
params = {'n_estimators': 206,  # 弱分类器的个数
          'max_depth': 3,  # 弱分类器（CART回归树）的最大深度
          'min_samples_split': 5,  # 分裂内部节点所需的最小样本数
          'learning_rate': 0.05,  # 学习率
          'loss': 'squared_error'}  # 损失函数类型
for i in range(1, 20):
    clf = GradientBoostingRegressor(**params)
    clf.fit(X, y)
    Y = clf.predict(X_predict)
    Y_pre = Y + Y_pre
    T = pow(10, 3.1529 - 0.3859 * ((Y_pre / 20) * 96.35 + 390.8) / 248.7) - 273.15

predict_data.insert(loc=len(predict_data.columns), column='T', value=T)
predict_data.to_csv('D:/FJW/newproject3/predict/new_A2BCX6_result.csv', index=False, sep=',')
