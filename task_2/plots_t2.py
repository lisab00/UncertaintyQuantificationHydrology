'''
Created on 19.11.2024

@author: lisa
'''

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

main_dir = Path(r'C:\Users\lihel\Documents\MMUQ_Python_Setup\MMUQ_Python_Setup\EclipsePortable426\Data\mmuq_ws2425\hmg\UncertaintyQuantificationHydrology')

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

# import optimised parameter vector from Task 1
iterations_df = pd.read_csv(main_dir / "outputs_task1" / "csv_outputs" / "output_one_per_iteration_tol_0.01_seed_123.csv")
iterations_df.columns = ["Obj_fct_value", "prm_value"]

# get best objective function value
true_ofv = iterations_df['Obj_fct_value'].iloc[-1]
df = pd.DataFrame(iterations_df['prm_value'])

df['prm_value'] = df['prm_value'].str.replace("[ ", "")
df['prm_value'] = df['prm_value'].str.replace("[", "")
df['prm_value'] = df['prm_value'].str.replace("]", "")
df['prm_value'] = df['prm_value'].str.split()
df['prm_value'] = [[float(p) for p in prm] for prm in df['prm_value']]

df[prm_names] = pd.DataFrame(df['prm_value'].tolist(), index=df.index)

# get best parameter vector
last_prm_values = df.iloc[-1]
last_prm_values = last_prm_values.drop('prm_value')

# impoer results from task 2
df_all_prm_changes = pd.read_csv(main_dir / 'task_2' / 'all_prm_changes_0.05.csv' , sep=';')
df_all_ofv_changes = pd.read_csv(main_dir / 'task_2' / 'all_ofv_changes_0.05.csv' , sep=';')
df_cumulated_output = pd.read_csv(main_dir / 'task_2' / 'cumulated_output_0.05.csv' , sep=';')

# # produce all plots

# plots for each prm
for prm_index in range(len(prm_names)):

    true_prm = last_prm_values.iloc[prm_index]
    prm_name = prm_names[prm_index]

    fig = plt.figure()

    plt.plot(df_all_prm_changes.iloc[: , prm_index], df_all_ofv_changes.iloc[: , prm_index])
    plt.axhline(y=true_ofv, color='orange', linestyle='--', linewidth=1.5, label='True objective value')
    plt.axvline(x=true_prm, color='green', linestyle='--', linewidth=1.5, label='True parameter value')

    plt.grid()
    plt.legend()
    plt.xticks(rotation=45)
    plt.xlabel('Parameter value')
    plt.ylabel('Objective function value')
    plt.title(f'Plot sensitivity of {prm_name}')

    fig.savefig(main_dir / 'task_2' / 'plots' / f'{prm_name}_sensitivity.png', bbox_inches='tight')

    plt.close(fig)

# scatter plot highest in normal scale
fig = plt.figure()

plt.scatter(df_cumulated_output['changed_parameter'], df_cumulated_output['new_obj_fct_value_after_change'])

plt.grid()
plt.legend()

plt.xticks(rotation=45)

plt.xlabel('Parameters')
plt.ylabel('Objective function values')

plt.title('Parameters vs. worst objective function values')

fig.savefig(main_dir / 'task_2' / 'plots' / 'scatter.png', bbox_inches='tight')
plt.close(fig)

# scatter plot highest in log scale
fig = plt.figure()

plt.scatter(df_cumulated_output['changed_parameter'], df_cumulated_output['new_obj_fct_value_after_change'])
plt.yscale('log')

plt.grid()
plt.legend()

plt.xticks(rotation=45)

plt.xlabel('Parameters')
plt.ylabel('Objective function values (log-scale)')

plt.title('Parameters vs. worst objective function values')

fig.savefig(main_dir / 'task_2' / 'plots' / 'scatter_log_scale.png', bbox_inches='tight')
plt.close(fig)
