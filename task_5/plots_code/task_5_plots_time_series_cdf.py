import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

main_dir = Path(r'C:\Users\lihel\Documents\MMUQ_Python_Setup\MMUQ_Python_Setup\EclipsePortable426\Data\mmuq_ws2425\hmg\UncertaintyQuantificationHydrology')

# read true time series
df = pd.read_csv(main_dir / "data" / "time_series__24163005" / "time_series___24163005.csv", sep=';')
print(df.head())

# read input data from task 5
df_pert_ddho = pd.read_csv(main_dir / "task_5" / "output" / "output_ddho.csv", index_col=0)

#=============================================================================
# plot cdf of perturbed ddho series
'''for i in np.arange(len(df_pert_ppts)):
    if i == 0:
        sns.kdeplot(df_pert_ddho.iloc[i], cumulative=True, alpha=0.5, color='orange', label='perturbed', linewidth=3)
    else:
        sns.kdeplot(df_pert_ddho.iloc[i], cumulative=True, alpha=0.5, color='orange', linewidth=3)

    plt.title("Compare original and perturbed ddho time series")
    plt.xlabel("ddho values")
    plt.ylabel("Cumulative distribution function")
    if i % 100 == 0:
        print(f"{i} plots created")

sns.kdeplot(df["ddho__ref"], cumulative=True, color="blue", label="original", linewidth=1)
plt.legend()
plt.show()'''

#=============================================================================
# plot cdf of ofv

iterations_df = pd.read_csv(main_dir / "task_1" / "outputs_task1" / "csv_outputs" / "output_one_per_iteration_tol_0.01_seed_123.csv")
iterations_df.columns = ["Obj_fct_value", "prm_value"]
'''ofv_true = iterations_df["Obj_fct_value"].iloc[-1]

# read input data from task 5
df_inp_ch = pd.read_csv(main_dir / "task_5" / "output" / "input_changes.csv", index_col=0)

sns.kdeplot(1 - df_inp_ch["new_opt_ofv"], cumulative=True, color="blue", label="recalibrated")
plt.axvline(1 - ofv_true, color="orange", linestyle="--", label="reference OFV")
plt.title("Reference NSE and recalibrated NSEs")
plt.xlabel("NSEs")
plt.ylabel("Cumulative distribution function")
plt.legend()
plt.show()'''

#=============================================================================
# plot prm cdfs

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

prm_val_true = iterations_df["prm_value"].iloc[-1]
df_inp_ch = pd.read_csv(main_dir / "task_5" / "output" / "input_changes.csv", index_col=0)

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

