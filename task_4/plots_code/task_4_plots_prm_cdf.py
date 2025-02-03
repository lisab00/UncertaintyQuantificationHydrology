'''
Created on 01.02.2025

@author: hfran
'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

obj_fct_values = pd.read_csv("C:\\Users\\hfran\\Documents\\Uni\\Master\\hydrology\\MMUQ_Python_Setup\\EclipsePortable426\\Data\\mmuq_ws2425\\hmg\\data\\task_1\\output_one_per_iteration_tol_0.01_seed_123.csv", header=None)

obj_fct_values.columns = ["Obj_fct_value", "prm_value"]

prm_val_true = obj_fct_values["prm_value"].iloc[-1]
prm_val_true = np.array(np.fromstring(prm_val_true.strip('[]'), sep=' '))
print(len(prm_val_true))

obj_fct_values = pd.read_csv("C:\\Users\\hfran\\Documents\\Uni\\Master\\hydrology\\MMUQ_Python_Setup\\EclipsePortable426\\Data\\mmuq_ws2425\\hmg\\data\\task_4\\input_changes.csv")

obj_fct_values.columns = ["index", "new_ofv_old_params", "prm_value", "new_opt_ofv"]
df = pd.DataFrame(obj_fct_values['prm_value'])
print(df.head())
prm_names = [
        'snw_dth',
        'snw_ast',
        'snw_amt',
        'snw_amf',
        'snw_pmf',

        'sl0_mse',
        'sl1_mse',

        'sl0_fcy',
        'sl0_bt0',

        'sl1_pwp',
        'sl1_fcy',
        'sl1_bt0',

        'urr_dth',
        'lrr_dth',

        'urr_rsr',
        'urr_tdh',
        'urr_tdr',
        'urr_cst',
        'urr_dro',
        'urr_ulc',

        'lrr_tdh',
        'lrr_cst',
        'lrr_dro',

    ]
print(len(prm_names))
print(print(df['prm_value'].iloc[0]))

df['prm_value'] = df['prm_value'].str.replace("[ ", "")
df['prm_value'] = df['prm_value'].str.replace("[", "")
df['prm_value'] = df['prm_value'].str.replace("]", "")
df['prm_value'] = df['prm_value'].str.split()
df['prm_value'] = [[float(p) for p in prm] for prm in df['prm_value']]

df[prm_names] = pd.DataFrame(df['prm_value'].tolist(), index=df.index)
print(df.head())
print(len(df[prm_names[0]]))
# create subplots for each prm
rows, cols = 5, 5
fig, axes = plt.subplots(rows, cols, figsize=(20, 15), constrained_layout=True)
print(len(prm_val_true))

for ax, i in zip(axes.flat, range(len(prm_names))):
    sns.kdeplot(df[prm_names[i]], cumulative=True, ax=ax, warn_singular=False)
    ax.axvline(x=prm_val_true[i], color='orange', linestyle='--', label='True Value')
    ax.set_title(f"{prm_names[i]}")
    ax.set_ylabel("CDF")
    ax.set_xlabel('')
    # ax.tick_params(axis='x')
    # ax.legend()

for ax in axes.flat[len(prm_val_true):]:
    ax.axis('off')

plt.show()

