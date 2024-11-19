'''
Created on 15.11.2024

@author: hfran
@author: lisa
'''

import os
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import csv

from collections import defaultdict

from hmg.models import hbv1d012a_py
from hmg.test import aa_run_model
from hmg import HBV1D012A

random.seed(123)

# main_dir = Path(r'C:\Users\hfran\Documents\Uni\Master\hydrology\MMUQ_Python_Setup\EclipsePortable426\Data\mmuq_ws2425\hmg\data')
main_dir = Path(r'C:\Users\lihel\Documents\MMUQ_Python_Setup\MMUQ_Python_Setup\EclipsePortable426\Data\mmuq_ws2425\hmg\UncertaintyQuantificationHydrology')
os.chdir(main_dir)

# Read input text time series as a pandas Dataframe object and
# cast the index to a datetime object.
# inp_dfe = pd.read_csv(r'time_series___24163005.csv', sep=';', index_col=0)
inp_dfe = pd.read_csv(main_dir / 'data' / 'time_series__24163005' / 'time_series___24163005.csv', sep=';', index_col=0)
inp_dfe.index = pd.to_datetime(inp_dfe.index, format='%Y-%m-%d-%H')

# Read the catchment area in meters squared. The first value is needed
# only.
# cca_srs = pd.read_csv(r'area___24163005.csv', sep=';', index_col=0)
cca_srs = pd.read_csv(main_dir / 'data' / 'time_series__24163005' / 'area___24163005.csv', sep=';', index_col=0)
ccaa = cca_srs.values[0, 0]

tems = inp_dfe.loc[:, 'tavg__ref'].values  # Temperature.
ppts = inp_dfe.loc[:, 'pptn__ref'].values  # Preciptiation.
pets = inp_dfe.loc[:, 'petn__ref'].values  # PET.
diso = inp_dfe.loc[:, 'diso__ref'].values  # Observed discharge.

tsps = tems.shape[0]  # Number of time steps.

# Conversion constant for mm/hour to m3/s.
dslr = ccaa / (3600 * 1000)  # For daily res. multiply denominator with 24.

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

bounds_dict = {
       'snw_dth': (0.00, 0.00),
        'snw_ast': (-1.0, +1.0),
        'snw_amt': (-0.0, +2.0),
        'snw_amf': (0.00, 2.00),
        'snw_pmf': (0.00, 2.00),

        'sl0_mse': (0.00, 1e+2),
        'sl1_mse': (0.00, 2e+2),

        'sl0_fcy': (0.00, 2e+2),
        'sl0_bt0': (0.00, 3.0),

        'sl1_pwp': (0.00, 4e+2),
        'sl1_fcy': (0.00, 4e+2),
        'sl1_bt0': (0.00, 4.00),

        'urr_dth': (0.00, 2e+1),
        'lrr_dth': (0.00, 5.00),

        'urr_rsr': (0.00, 1.00),
        'urr_tdh': (0.00, 1e+2),
        'urr_tdr': (0.00, 1.00),
        'urr_cst': (0.00, 1.00),
        'urr_dro': (0.00, 1.00),
        'urr_ulc': (0.00, 1.00),

        'lrr_tdh': (0.00, 1e+4),
        'lrr_cst': (0.00, 1.00),
        'lrr_dro': (0.00, 1.00),
        }

# import optimised parameter vector from Task 1
# iterations_df = pd.read_csv(main_dir / "task_1" / "output_one_per_iteration_tol_0.01_seed_123.csv")
iterations_df = pd.read_csv(main_dir / "outputs_task1" / "csv_outputs" / "output_one_per_iteration_tol_0.01_seed_123.csv")
iterations_df.columns = ["Obj_fct_value", "prm_value"]
# get best objective function value
last_objective_function = iterations_df['Obj_fct_value'].iloc[-1]
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


def nse(sim, obs):
    '''Nash-Suttcliffe Efficiency'''
    # return 1-nse ne nse to make minimization work
    return 1 - (1 - (np.sum((obs - sim) ** 2) / np.sum((obs - np.mean(obs)) ** 2)))


def objective_function_value(prms, modl_objt, metric: str, diso):
    # # read out metric string
    # Dictionary mapping string names to metric functions
    metrics = {
        # "sse": sse,
        # "rmse": rmse,
        "nse": nse,
        # "lognse": lognse,
        # "pbias": pbias
    }

    # Get the right metric function using dictionary
    metric_fun = metrics.get(metric)

    if not metric_fun:
        raise ValueError(f"Function '{metric}' is not recognized.")

    # # simulate model with prms and compute obj value
    # Pass the current parameters to model object
    modl_objt.set_parameters(prms)

    # Tell the model object that the simulation is not an optimization.
    modl_objt.set_optimization_flag(0)

    # Run the model for the given inputs, constants and parameters.
    modl_objt.run_model()

    # compute obj value with current parms
    diss = modl_objt.get_discharge()
    result = metric_fun(diss, diso)
    return result


def change_parameter_vector(i, last_prm_value, change_value):
    last_prm_value.iloc[i] = last_prm_value.iloc[i] * change_value

    # check if new param value is in bounds!
    # if not set to upper / lower bound

    bound_for_param = bounds_dict[prm_names[i]]

    if last_prm_value.iloc[i] > bound_for_param[1]:
        last_prm_value.iloc[i] = bound_for_param[1]

    elif last_prm_value.iloc[i] < bound_for_param[0]:
        last_prm_value.iloc[i] = bound_for_param[0]

    return last_prm_value

'''the keys of this dicts are the indices of the parameters
the values are the new parameter after changing (_prm_changes), and the ofv
that the changed prm produces (_ofv_changes).
per param index we will plot these against each other'''
all_perc_prm_changes = defaultdict(list)
all_perc_ofv_changes = defaultdict(list)


def change_all_params(last_prm_value, change_value, metric):

    for i in range(0, len(last_prm_value)):

        # copy of orig prms, bc we want to keep original last_prm_value for next iteration
        updated_prms = last_prm_value.copy()

        # update chosen parm in the vector
        updated_prms = change_parameter_vector(i, updated_prms, change_value)

        # compute ofv with new set of prm
        new_obj_function = objective_function_value(np.array(updated_prms.values, dtype=float), modl_objt, metric, diso)

        # store
        all_perc_prm_changes[i].append(updated_prms[i])  # append new value for param i
        all_perc_ofv_changes[i].append(new_obj_function)  # append new ofv for param i


if __name__ == "__main__":
    modl_objt = HBV1D012A()
    modl_objt.set_inputs(tems, ppts, pets)
    modl_objt.set_outputs(tsps)
    modl_objt.set_discharge_scaler(dslr)
    otps_lbls = modl_objt.get_output_labels()
    metric = "nse"

    # create loop to try out percentual changes between [-20%,+20%] in steps of 0.5%
    change_values = np.arange(0.8, 1.2 + 0.005, 0.005)

    for change_value in change_values:
        change_all_params(last_prm_values, change_value, metric)

    # store data in csv to use for plots
    df_prms = pd.DataFrame.from_dict(all_perc_prm_changes)
    df_prms.to_csv(main_dir / 'task_2' / 'all_prm_changes.csv', index=False, sep=';')
    df_ofv = pd.DataFrame.from_dict(all_perc_ofv_changes)
    df_ofv.to_csv(main_dir / 'task_2' / 'all_ofv_changes.csv', index=False, sep=';')

    # create one output dict with the most significant change per parameter
    cumulated_output = []

    for i in range(0, len(last_prm_values)):

            # determine which percentual prm change caused the greatest change in ofv
            # we can pick largest value bec. new ofv will never be better, i.e. lower, than best ofv
            greatest_ofv = max(all_perc_ofv_changes[i])
            greatest_change_index = all_perc_ofv_changes[i].index(greatest_ofv)

            all_output_dict = {
                'changed_parameter': prm_names[i],
                'param_mult_by': change_values[greatest_change_index],
                'old_param_value': last_prm_values.iloc[i],
                'new_param_value': all_perc_prm_changes[i][greatest_change_index],
                'param_bounds': bounds_dict[prm_names[i]],
                'old_best_obj_fct_value': last_objective_function,
                'new_obj_fct_value_after_change': greatest_ofv,
                'change_of_obj_fct_new/old': greatest_ofv / last_objective_function,
                }

            cumulated_output.append(all_output_dict)

    df = pd.DataFrame.from_dict(cumulated_output)
    df.to_csv(main_dir / 'task_2' / 'cumulated_output.csv', index=False, sep=';')
    print(f"Output printed to: {main_dir / 'task_2' / 'cumulated_output.csv'}")
