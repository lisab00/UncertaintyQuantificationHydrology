'''
Created on 12.12.2024

@author: hfran
'''

# correlation plots
# read in all optimization values, all iterations
# one dot = iteration n , param A vs param B

import os
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from hmg.models import hbv1d012a_py
from hmg.test import aa_run_model
from hmg import HBV1D012A

np.random.seed(123)

main_dir = Path(r'C:\Users\hfran\Documents\Uni\Master\hydrology\MMUQ_Python_Setup\EclipsePortable426\Data\mmuq_ws2425\hmg\data')
# main_dir = Path(r'C:\Users\lihel\Documents\MMUQ_Python_Setup\MMUQ_Python_Setup\EclipsePortable426\Data\mmuq_ws2425\hmg\UncertaintyQuantificationHydrology')
os.chdir(main_dir)

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

bounds_dict = {  # new global prm bounds on moodle
        'snw_dth': (0.00, 10.0),
        'snw_ast': (-1.0, +1.0),
        'snw_amt': (-0.0, +2.0),
        'snw_amf': (0.00, 2.00),
        'snw_pmf': (0.00, 2.00),

        'sl0_mse': (0.00, 1e+2),
        'sl1_mse': (0.00, 2e+2),

        'sl0_fcy': (5.00, 4e+1),
        'sl0_bt0': (1.00, 6.00),

        'sl1_pwp': (1.00, 2e+2),
        'sl1_fcy': (1e+2, 4e+2),
        'sl1_bt0': (1.00, 3.00),

        'urr_dth': (0.00, 2e+1),
        'lrr_dth': (0.00, 5.00),

        'urr_rsr': (0.00, 1.00),
        'urr_tdh': (0.00, 1e+2),
        'urr_tdr': (0.00, 1.00),
        'urr_cst': (1e-4, 1.00),
        'urr_dro': (1.00, 1.00),
        'urr_ulc': (0.00, 1.00),

        'lrr_tdh': (5e+2, 1e+4),
        'lrr_cst': (0.00, 1.00),
        'lrr_dro': (0.00, 1.00),
        }

# import optimised parameter vector from Task 1
iterations_df = pd.read_csv(main_dir / "task_1" / "output_all_evaluations_tol_0.01_seed_123.csv")
# iterations_df = pd.read_csv(main_dir / "task_1" / "outputs_task1" / "csv_outputs" / "output_one_per_iteration_tol_0.01_seed_123.csv")
iterations_df.columns = ["Obj_fct_value", "prm_value"]
# get best objective function value
# last_objective_function = iterations_df['Obj_fct_value'].iloc[-1]
iterations_df = iterations_df[iterations_df['Obj_fct_value'] < 0.1]
print(len(iterations_df))
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

for i in range(len(prm_names)):
    for j in range(len(prm_names)):
        if i != j:
            prm = prm_names[i]
            prm_two = prm_names[j]
            fig = plt.figure()

            plt.scatter(df[prm], df[prm_two], alpha=0.5, s=0.5)

            plt.grid()
            plt.legend()

            plt.xticks(rotation=45)

            plt.xlabel(f'Parameter value of {prm}')
            plt.ylabel(f'Parameter value of {prm_two}')
            # plt.yscale('log')

            plt.title('Checking for Correlation')

            fig.savefig(f'C:\\Users\\hfran\\Documents\\Uni\\Master\\hydrology\\MMUQ_Python_Setup\\EclipsePortable426\\Data\\mmuq_ws2425\\hmg\\data\\task_1\\below_01_correlation_{prm}_vs_{prm_two}.png', bbox_inches='tight')
            plt.close(fig)

