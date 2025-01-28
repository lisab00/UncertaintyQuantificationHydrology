'''
created on 25.01.25
@author lisa
plots for task 4
'''

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

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

main_dir = Path(r'C:\Users\lihel\Documents\MMUQ_Python_Setup\MMUQ_Python_Setup\EclipsePortable426\Data\mmuq_ws2425\hmg\UncertaintyQuantificationHydrology')

# import true prm vector from task 1
iterations_df = pd.read_csv(main_dir / "task_1" / "outputs_task1" / "csv_outputs" / "output_one_per_iteration_tol_0.01_seed_123.csv")
iterations_df.columns = ["Obj_fct_value", "prm_value"]
prm_val_true = iterations_df["prm_value"].iloc[-1]

# read input data from task 4
df_inp_ch = pd.read_csv(main_dir / "task_4" / "output" / "input_changes.csv", index_col=0)

# bring prm_values into correct format
prm_val_new = np.array([np.fromstring(row.strip('[]'), sep=' ') for row in df_inp_ch["new_opt_params"]]).T.tolist()

# create subplots for each prm
rows, cols = 5, 5
fig, axes = plt.subplots(rows, cols, figsize=(20, 15), constrained_layout=True)

for ax, i in zip(axes.flat, np.arange(len(prm_val_new))):
    sns.kdeplot(prm_val_new[i], cumulative=True, ax=ax, warn_singular=False)
    ax.axvline(x=prm_val_true[i], color='orange', linestyle='--', label='True Value')
    ax.set_title(f"{prm_names[i]}")
    ax.set_ylabel("CDF")
    ax.tick_params(axis='x', labelbottom=False)
    # ax.legend()

for ax in axes.flat[len(prm_val_new):]:
    ax.axis('off')

plt.show()
