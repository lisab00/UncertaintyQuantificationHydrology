'''
created on 28.01.25
@author lisa
plots for task 4
'''

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

main_dir = Path(r'C:\Users\lihel\Documents\MMUQ_Python_Setup\MMUQ_Python_Setup\EclipsePortable426\Data\mmuq_ws2425\hmg\UncertaintyQuantificationHydrology')

# import data from previous task
iterations_df = pd.read_csv(main_dir / "task_1" / "outputs_task1" / "csv_outputs" / "output_one_per_iteration_tol_0.01_seed_123.csv")
iterations_df.columns = ["Obj_fct_value", "prm_value"]
true_data = pd.read_csv(main_dir / 'data' / 'time_series__24163005' / 'time_series___24163005.csv', sep=';')

# read input data from task 4
df_inp_ch = pd.read_csv(main_dir / "task_4" / "output" / "input_changes.csv")

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
df_pert_ppts = pd.read_csv(main_dir / "task_4" / "output" / "output_ppts.csv", index_col=0)
df_pert_tems = pd.read_csv(main_dir / "task_4" / "output" / "output_tems.csv", index_col=0)

# reference data and mean data
ofv_ref_task1 = 1 - iterations_df["Obj_fct_value"].iloc[-1]
mean_ppts_ref = np.mean(true_data["pptn__ref"])
mean_tem_ref = np.mean(true_data["tavg__ref"])

means_ppts = []
means_tems = []
for i in np.arange(len(df_pert_ppts)):
    means_ppts.append(np.mean(df_pert_ppts.iloc[i]))
    means_tems.append(np.mean(df_pert_tems.iloc[i]))

# plot ppts not recal
'''fig = plt.figure()
plt.scatter(means_ppts, ofv_old)
plt.scatter(mean_ppts_ref, ofv_ref_task1, color="orange", s=50, label="reference")
plt.xlabel("Mean of ppts time series")
plt.ylabel("Perturbed NSE")
plt.title("Mean of ppts time series against perturbed NSE values")
plt.legend()
plt.show()
fig.savefig(main_dir / 'task_4' / 'plots' / 'scatter_ppts_mean_ofv.png', bbox_inches='tight')
plt.close(fig)'''

# plot ppts recal
'''fig = plt.figure()
plt.scatter(means_ppts, ofv_recal)
plt.scatter(mean_ppts_ref, ofv_ref_task1, color="orange", s=50, label="reference")
plt.xlabel("Mean of ppts time series")
plt.ylabel("Recalibrated NSE")
plt.title("Mean of ppts time series against recalibrated NSE values")
plt.legend()
plt.show()
fig.savefig(main_dir / 'task_4' / 'plots' / 'scatter_ppts_mean_ofv_recal.png', bbox_inches='tight')
plt.close(fig)'''

# plot tems not recal
'''fig = plt.figure()
plt.scatter(means_tems, ofv_old)
plt.scatter(mean_tem_ref, ofv_ref_task1, color="orange", s=50, label="reference")
plt.xlabel("Mean of temp time series")
plt.ylabel("Perturbed NSE")
plt.title("Mean of temp time series against perturbed NSE values")
plt.legend()
plt.show()
fig.savefig(main_dir / 'task_4' / 'plots' / 'scatter_tems_mean_ofv.png', bbox_inches='tight')
plt.close(fig)'''

# plot tems recal
'''fig = plt.figure()
plt.scatter(means_tems, ofv_recal)
plt.scatter(mean_tem_ref, ofv_ref_task1, color="orange", s=50, label="reference")
plt.xlabel("Mean of temp time series")
plt.ylabel("Recalibrated NSE")
plt.title("Mean of temp time series against recalibrated NSE values")
plt.legend()
plt.show()
fig.savefig(main_dir / 'task_4' / 'plots' / 'scatter_tems_mean_ofv_recal.png', bbox_inches='tight')
plt.close(fig)'''
