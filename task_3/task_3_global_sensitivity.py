'''
Created on 29.11.2024

@author: hfran
'''

# goal: compute sobol indices

# generate random numbers distributed uniformly according to their param bounds

# Matrix A and B are sampled uniformly from param bounds

# Matrix C is a mixup of A and B

# for every row, compute the obj fct value

# scatterplots: sample uniformly from param bounds = x-axis, vs y obj.fct.value

import os
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from hmg.models import hbv1d012a_py
from hmg.test import aa_run_model
from hmg import HBV1D012A

np.random.seed(123)

# main_dir = Path(r'C:\Users\hfran\Documents\Uni\Master\hydrology\MMUQ_Python_Setup\EclipsePortable426\Data\mmuq_ws2425\hmg\data')
main_dir = Path(r'C:\Users\lihel\Documents\MMUQ_Python_Setup\MMUQ_Python_Setup\EclipsePortable426\Data\mmuq_ws2425\hmg\UncertaintyQuantificationHydrology')
os.chdir(main_dir)

# Read input text time series as a pandas Dataframe object and
# cast the index to a datetime object.
# inp_dfe = pd.read_csv(r'time_series___24163005.csv', sep=';', index_col=0)
inp_dfe = pd.read_csv(r'data\time_series__24163005\time_series___24163005.csv', sep=';', index_col=0)
inp_dfe.index = pd.to_datetime(inp_dfe.index, format='%Y-%m-%d-%H')

# Read the catchment area in meters squared. The first value is needed
# only.
# cca_srs = pd.read_csv(r'area___24163005.csv', sep=';', index_col=0)
cca_srs = pd.read_csv(r'data\time_series__24163005\area___24163005.csv', sep=';', index_col=0)
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
# iterations_df = pd.read_csv(main_dir / "task_1" / "output_one_per_iteration_tol_0.01_seed_123.csv")
iterations_df = pd.read_csv(main_dir / "task_1" / "outputs_task1" / "csv_outputs" / "output_one_per_iteration_tol_0.01_seed_123.csv")
iterations_df.columns = ["Obj_fct_value", "prm_value"]
# get best objective function value
last_objective_function = iterations_df['Obj_fct_value'].iloc[-1]
print(last_objective_function)
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
print(last_prm_values)


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
    print(last_prm_value.iloc[i])
    last_prm_value.iloc[i] = last_prm_value.iloc[i] * change_value
    print(last_prm_value.iloc[i])
    # check if new param value is in bounds!
    # if not set to upper / lower bound

    bound_for_param = bounds_dict[prm_names[i]]

    if last_prm_value.iloc[i] > bound_for_param[1]:
        last_prm_value.iloc[i] = bound_for_param[1]

    elif last_prm_value.iloc[i] < bound_for_param[0]:
        last_prm_value.iloc[i] = bound_for_param[0]

    return last_prm_value.iloc[i]


def generate_sobol_samples(dim, num_samples):

    num_cols = dim
    matrix = np.zeros((num_samples, num_cols))

    for i, (low, high) in enumerate(bounds_dict.values()):
        matrix[:, i] = np.random.uniform(low, high, size=num_samples)

    return matrix


def compute_sobol_indices(model, num_samples, dim):
    # Generate Sobol samples
    A = generate_sobol_samples(dim, num_samples)
    B = generate_sobol_samples(dim, num_samples)

    Y_A = np.zeros(num_samples)
    Y_B = np.zeros(num_samples)

    # Evaluate the model at the generated samples
    for i in range(num_samples):
        y_A = model(A[i,:], modl_objt, metric, diso)
        y_B = model(B[i,:], modl_objt, metric, diso)
        Y_A[i] = y_A
        Y_B[i] = y_B

    # Calculate the mean of the model output
    f02 = np.mean(Y_A) * np.mean(Y_B)

    # First-order and total-order Sobol indices
    S_first = np.zeros(dim)
    S_total = np.zeros(dim)

    for i in range(dim):
        # Create matrices B_Ai (swapping columns)
        B_Ai = np.copy(B)
        B_Ai[:, i] = A[:, i]
        Y_BAi = np.zeros(num_samples)

        # Evaluate the function for the modified matrices
        for j in range(num_samples):
            y_BAi = model(B_Ai[j,:], modl_objt, metric, diso)
            Y_BAi[j] = y_BAi

        # First-order index calculation
        S_first[i] = (np.mean(Y_A * Y_BAi) - f02) / (np.mean(Y_A * Y_A) - f02)
        print(Y_B)

        # Total-order index calculation
        S_total[i] = 1 - (np.mean(Y_B * Y_BAi) - f02) / (np.mean(Y_A * Y_A) - f02)

    return S_first, S_total


def plot_sobol_indices(S_first, S_total):
    parameters = prm_names  # Parameter names

    # Plotting
    x = np.arange(len(parameters))  # The label locations
    width = 0.35  # Width of the bars

    fig, ax = plt.subplots(figsize=(8, 6))
    bars1 = ax.bar(x - width / 2, S_first, width, label='First-order Index')
    bars2 = ax.bar(x + width / 2, S_total, width, label='Total-effect Index')

    # Add some text for labels, title, and custom x-axis tick labels, etc.
    ax.set_xlabel('Parameters')
    ax.set_ylabel('Sobol Indices')
    ax.set_title('Sobol Indices for Sensitivity Analysis')
    ax.set_xticks(x)
    ax.set_xticklabels(parameters)
    plt.xticks(rotation=90)
    ax.legend()

    # Display the plot
    plt.show()

    # Save plot
    # fig.savefig(main_dir / 'task_3' / 'plots' / 'sobol_indices', bbox_inches='tight')


if __name__ == "__main__":
    modl_objt = HBV1D012A()
    modl_objt.set_inputs(tems, ppts, pets)
    modl_objt.set_outputs(tsps)
    modl_objt.set_discharge_scaler(dslr)
    otps_lbls = modl_objt.get_output_labels()
    metric = "nse"
    print(bounds_dict.values)
    print(type(bounds_dict.values))

    num_samples = 20
    dim = len(prm_names)
    S_first, S_total = compute_sobol_indices(objective_function_value, num_samples, dim)
    print(S_first, S_total)
    plot_sobol_indices(S_first, S_total)
