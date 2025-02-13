'''
Created on 01.02.2025

@author: hfran
'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
import matplotlib.ticker as mticker

obj_fct_values = pd.read_csv("C:\\Users\\hfran\\Documents\\Uni\\Master\\hydrology\\MMUQ_Python_Setup\\EclipsePortable426\\Data\\mmuq_ws2425\\hmg\\data\\task_1\\output_one_per_iteration_tol_0.01_seed_123.csv", header=None)

obj_fct_values.columns = ["Obj_fct_value", "prm_value"]

prm_val_true = obj_fct_values["prm_value"].iloc[-1]
ofv_ref_task1 = 1 - obj_fct_values["Obj_fct_value"].iloc[-1]

prm_val_true = np.array(np.fromstring(prm_val_true.strip('[]'), sep=' '))
print(len(prm_val_true))

obj_fct_values = pd.read_csv("C:\\Users\\hfran\\Documents\\Uni\\Master\\hydrology\\MMUQ_Python_Setup\\EclipsePortable426\\Data\\mmuq_ws2425\\hmg\\data\\task_4\\input_changes_redone.csv")

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
'''# create subplots for each prm
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
fig.savefig(f'C:\\Users\\hfran\\Documents\\Uni\\Master\\hydrology\\MMUQ_Python_Setup\\EclipsePortable426\\Data\\mmuq_ws2425\\hmg\\data\\task_4\\cdf_prms.svg', bbox_inches='tight')
'''

true_data = pd.read_csv("C:\\Users\\hfran\\Documents\\Uni\\Master\\hydrology\\MMUQ_Python_Setup\\EclipsePortable426\\Data\\mmuq_ws2425\\hmg\\data\\time_series___24163005.csv", sep=';')

# read input data from task 4
df_inp_ch = obj_fct_values

# ofv time series
ofv_old = 1 - df_inp_ch["new_ofv_old_params"]
ofv_recal = 1 - df_inp_ch["new_opt_ofv"]

# ## scatter ofvs against recalibrated ofvs
'''fig = plt.figure()
plt.scatter(ofv_old, ofv_recal)
plt.title("Compare perturbed against recalibrated NSE")
plt.xlabel("Perturbed NSE before recalibration")
plt.ylabel("Recalibrated NSE")
plt.show()
fig.savefig(main_dir / 'task_4' / 'plots' / 'ofv_scatter_old_recal.png', bbox_inches='tight')
plt.close(fig)'''

# ## scatter mean perturbed time series against ofv

# read time series and create mean data
df_pert_ppts = pd.read_csv("C:\\Users\\hfran\\Documents\\Uni\\Master\\hydrology\\MMUQ_Python_Setup\\EclipsePortable426\\Data\\mmuq_ws2425\\hmg\\data\\task_4\\output_ppts_redone.csv", index_col=0)
df_pert_tems = pd.read_csv("C:\\Users\\hfran\\Documents\\Uni\\Master\\hydrology\\MMUQ_Python_Setup\\EclipsePortable426\\Data\\mmuq_ws2425\\hmg\\data\\task_4\\output_tems_redone.csv", index_col=0)

main_dir = Path(r"C:\\Users\\hfran\\Documents\\Uni\\Master\\hydrology\\MMUQ_Python_Setup\\EclipsePortable426\\Data\\mmuq_ws2425\\hmg\\data")
# reference data and mean data
mean_ppts_ref = np.mean(true_data["pptn__ref"])
mean_tem_ref = np.mean(true_data["tavg__ref"])

means_ppts = []
means_tems = []
for i in np.arange(len(df_pert_ppts)):
    means_ppts.append((np.mean(df_pert_ppts.iloc[i]) - mean_ppts_ref) / mean_ppts_ref)
    means_tems.append(np.mean(df_pert_tems.iloc[i]) - mean_tem_ref)

fmt = '%.0f%%'
# plot ppts not recal
fig = plt.figure()
plt.scatter(means_ppts, ofv_old, alpha=0.5)
plt.scatter(0, ofv_ref_task1, color="orange", s=50, label="reference")
plt.xlabel("Mean relative change of ppts time series")
plt.ylabel("Perturbed NSE")
plt.title("Mean change of ppts time series against perturbed NSE values")
plt.legend()
ax = plt.gca()
ax.xaxis.set_major_formatter(mticker.PercentFormatter(1))
plt.show()
fig.savefig(main_dir / 'task_4' / 'plots' / 'scatter_ppts_mean_ofv.png', bbox_inches='tight')
plt.close(fig)

# plot ppts recal
fig = plt.figure()
plt.scatter(means_ppts, ofv_recal, alpha=0.5)
plt.scatter(0, ofv_ref_task1, color="orange", s=50, label="reference")
plt.xlabel("Mean relative change of ppts time series")
plt.ylabel("Recalibrated NSE")
plt.title("Mean change of ppts time series against recalibrated NSE values")
plt.legend()
ax = plt.gca()
ax.xaxis.set_major_formatter(mticker.PercentFormatter(1))
plt.show()
fig.savefig(main_dir / 'task_4' / 'plots' / 'scatter_ppts_mean_ofv_recal.png', bbox_inches='tight')
plt.close(fig)

# plot tems not recal
fig = plt.figure()
plt.scatter(means_tems, ofv_old, alpha=0.5)
plt.scatter(0, ofv_ref_task1, color="orange", s=50, label="reference")
plt.xlabel("Mean change of temp time series")
plt.ylabel("Perturbed NSE")
plt.title("Mean change of temp time series against perturbed NSE values")
plt.legend()
plt.show()
fig.savefig(main_dir / 'task_4' / 'plots' / 'scatter_tems_mean_ofv.png', bbox_inches='tight')
plt.close(fig)
# plot tems recal
fig = plt.figure()
plt.scatter(means_tems, ofv_recal, alpha=0.5)
plt.scatter(0, ofv_ref_task1, color="orange", s=50, label="reference")
plt.xlabel("Mean change of temp time series")
plt.ylabel("Recalibrated NSE")
plt.title("Mean change of temp time series against recalibrated NSE values")
plt.legend()
plt.show()
fig.savefig(main_dir / 'task_4' / 'plots' / 'scatter_tems_mean_ofv_recal.png', bbox_inches='tight')
plt.close(fig)
