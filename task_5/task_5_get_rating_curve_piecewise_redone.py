'''
Created on 15.11.2024
@author: hfran
'''

import os
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import csv

from collections import defaultdict
from scipy.optimize import differential_evolution
from scipy.optimize import curve_fit

from hmg.models import hbv1d012a_cy
from hmg.test import aa_run_model
from hmg import HBV1D012A

random.seed(123)

main_dir = Path(r'C:\Users\hfran\Documents\Uni\Master\hydrology\MMUQ_Python_Setup\EclipsePortable426\Data\mmuq_ws2425\hmg\data')
# main_dir = Path(r'C:\Users\lihel\Documents\MMUQ_Python_Setup\MMUQ_Python_Setup\EclipsePortable426\Data\mmuq_ws2425\hmg\UncertaintyQuantificationHydrology')
os.chdir(main_dir)

# Read input text time series as a pandas Dataframe object and
# cast the index to a datetime object.
inp_dfe = pd.read_csv(r'time_series___24163005.csv', sep=';', index_col=0)
# inp_dfe = pd.read_csv(main_dir / 'data' / 'time_series__24163005' / 'time_series___24163005.csv', sep=';', index_col=0)
inp_dfe.index = pd.to_datetime(inp_dfe.index, format='%Y-%m-%d-%H')

# Read the catchment area in meters squared. The first value is needed only.
cca_srs = pd.read_csv(r'area___24163005.csv', sep=';', index_col=0)
# cca_srs = pd.read_csv(main_dir / 'data' / 'time_series__24163005' / 'area___24163005.csv', sep=';', index_col=0)
ccaa = cca_srs.values[0, 0]

tems = inp_dfe.loc[:, 'tavg__ref'].values  # Temperature.
ppts = inp_dfe.loc[:, 'pptn__ref'].values  # Preciptiation.
pets = inp_dfe.loc[:, 'petn__ref'].values  # PET.
diso = inp_dfe.loc[:, 'diso__ref'].values  # Observed discharge.
ddho = inp_dfe.loc[:, 'ddho__ref'].values  # Observed discharge.

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
iterations_df = pd.read_csv(main_dir / "task_1" / "output_one_per_iteration_tol_0.01_seed_123.csv")
# iterations_df = pd.read_csv(main_dir / "outputs_task1" / "csv_outputs" / "output_one_per_iteration_tol_0.01_seed_123.csv")
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

'''the keys of this dicts are the indices of the parameters
the values are the new parameter after changing (_prm_changes), and the ofv
that the changed prm produces (_ofv_changes).
per param index we will plot these against each other'''
all_perc_prm_changes = defaultdict(list)
all_perc_ofv_changes = defaultdict(list)

change_value_tems = []
change_value_ppts = []
new_ofv = []
new_opt_params = []
new_opt_ofv = []
tems_vector = []
ppts_vector = []


def change_series_and_compute_ofv(diso, metric="nse"):

    modl_objt = HBV1D012A()
    modl_objt.set_inputs(tems, ppts, pets)
    modl_objt.set_outputs(tsps)
    modl_objt.set_discharge_scaler(dslr)
    otps_lbls = modl_objt.get_output_labels()

    # compute ofv with new set of prm
    new_obj_function = objective_function_value(np.array(last_prm_values.values, dtype=float), modl_objt, metric, diso)

    # store

    new_ofv.append(new_obj_function)

    # optimize parameters again
    res = differential_evolution(func=objective_function_value,  # function to be minimized
                                 args=(modl_objt, metric, diso),  # fixed args for func
                                 bounds=list(bounds_dict.values()),  # bounds on prms
                                 # maxiter=100,  # max number of iterations to be performed
                                 # callback=callback,  # write intermediate values to csv file
                                 tol=0.01,  # allow for early stopping
                                 seed=123,  # make stochastic minimization reproducible
                                 disp=True,  # print intermediate results
                                 polish=False)  # always set this to false

    # obtain fitted prms
    res_prms = res.x
    res_suc = res.success
    res_fun_val = 1 - res.fun
    new_opt_params.append(res_prms)
    new_opt_ofv.append(res.fun)

    # print outputs
    print(f"optimized prms: {res_prms}")
    print(f"success: {res_suc}")
    print(f"value of performance metric: {res_fun_val}")
    print(f"message: {res.message}")
    print(f"number of iterations performed: {res.nit}")


h0_initial = min(ddho)  # - 5


def power_eq(h, a, b, c, d, e, f, g, i):

    return a + b * h + c * h ** 2 + d * h ** 3 + e * h ** 4 + f * h ** 5 + g * h ** 6 + i * h ** 7  # a * np.power((h - h0_initial), b)


def fit_power_eq():
    # Curve fitting to estimate a, b, and h0
    h0_initial = min(ddho)
    ddho_new = ddho - h0_initial
    popt, pcov = curve_fit(power_eq, ydata=diso, xdata=ddho_new)

    # Extract fitted parameters

    # print(f"Fitted Parameters: a={a:.2f}, b={b:.2f}")  # h0={h0:.2f}")

    # Predict discharges using the fitted curve
    fitted_discharges = power_eq(ddho_new, *popt)

    # Plot the results
    plt.scatter(ddho_new, diso, label="Observed Data", color="blue", alpha=0.2)
    plt.plot(ddho_new, fitted_discharges, label="Fitted Power Equation", color="red")
    plt.xlabel("Depth (h)")
    plt.ylabel("Discharge (Q)")
    plt.legend()
    plt.title("Fitted Power Equation to Depth-Discharge Data")
    plt.show()
    return popt, fitted_discharges


save_change_vector = []


def change_ddho(ddho):
    n = len(ddho)
    change_vector = np.random.uniform(-5, 5, n)
    save_change_vector.append(change_vector)
    ddho_new = ddho + change_vector
    return ddho_new


def mae(sim, obs):
    return max(np.abs(sim - obs))


def compute_disp(ddho, coeffs):
    '''evaluate power function on ddho and return predicted discharge disp
    input should be perturbed ddho
    '''
    # i = (np.abs(ddho - 432)).argmin()
    indices_smaller = np.where(ddho < 432)
    indices_bigger = np.where(ddho >= 432)

    # Initialize an array to store results
    disp = np.zeros_like(ddho, dtype=float)
    disp[indices_smaller] = np.polyval(coeffs[0], ddho[indices_smaller])
    # print(len(disp_s1))
    disp[indices_bigger] = np.polyval(coeffs[1], ddho[indices_bigger])
    # print(len(disp_s2))

    # disp_s2_concate = np.delete(disp_s2, 0)
    # print(len(disp_s2_concate))
    # disp = np.concatenate((disp_s1, disp_s2))
    return disp


def compute_coeffs(ddho, diso, degree_s1, degree_s2):
    '''returns coeffs for rating curve
    inputs are original ddho / diso time series
    put i as: i = (np.abs(ddho - 432)).argmin()
    '''
    i = (np.abs(ddho - 432)).argmin()

    # define intervals for fitted curves
    ddho_s1 = ddho[:i + 1]
    diso_s1 = diso[:i + 1]
    diso_s2 = diso[i:]
    ddho_s2 = ddho[i:]

    # fit curves for each interval
    coeffs_s1 = np.polyfit(ddho_s1, diso_s1, deg=degree_s1)
    coeffs_s2 = np.polyfit(ddho_s2, diso_s2, deg=degree_s2)

    return (coeffs_s1, coeffs_s2)


if __name__ == "__main__":

    np.random.seed(123)
    # popt, fitted_discharges = fit_power_eq()
    # print(popt)

    # transform data (required)
    # ensure ddho is sorted (required for piecewise interpolation)
    sorted_indices = np.argsort(ddho)
    ddho = ddho[sorted_indices]
    diso = diso[sorted_indices]

    # remove outliers from dataset
    residuals = abs(diso - np.polyval(np.polyfit(ddho, diso, deg=12), ddho))
    res_sorted_indices = np.argsort(-residuals)
    num_outliers = 55  # visual analysis
    outlier_indices = res_sorted_indices[:num_outliers]
    ddho = np.delete(ddho, outlier_indices)
    diso = np.delete(diso, outlier_indices)
    # tems = np.delete(tems, outlier_indices)
    # ppts = np.delete(ppts, outlier_indices)
    # pets = np.delete(pets, outlier_indices)

    tsps = tems.shape[0]
    print(tsps)
    print(len(diso))
    print(len(tems))

    # decide on degree and index for power function. do this once to obtain coeffs
    degree_s1 = 5
    degree_s2 = 19
    coeffs = compute_coeffs(ddho, diso, degree_s1, degree_s2)
    # print(coeffs)

    tems = inp_dfe.loc[:, 'tavg__ref'].values  # Temperature.
    ppts = inp_dfe.loc[:, 'pptn__ref'].values  # Preciptiation.
    pets = inp_dfe.loc[:, 'petn__ref'].values  # PET.
    diso = inp_dfe.loc[:, 'diso__ref'].values  # Observed discharge.
    ddho = inp_dfe.loc[:, 'ddho__ref'].values  # Observed discharge.

    save_perturbed_ddho = []
    for k in range(2000):
        ddho_perturbed = change_ddho(ddho)
        # ddho_perturbed = np.delete(ddho, outlier_indices)
        # run this every loop
        # call compute_disp for evers perturbed time series with same coeffs
        # make sure you perturb time series where the outliers are removed!
        disp = compute_disp(ddho_perturbed, coeffs)
        print(len(disp))
        print(len(ddho_perturbed))
        save_perturbed_ddho.append(ddho_perturbed)
        # fitted_discharges = power_eq(np.maximum(ddho_perturbed - h0_initial, 0), *popt)
        change_series_and_compute_ofv(disp, metric="nse")

    # store data in csv to use for plots
    output_df = pd.DataFrame({'new_ofv_old_params': new_ofv,
                               'new_opt_params': new_opt_params,
                               'new_opt_ofv': new_opt_ofv})
    # print(len(output_df['change_value_ppts'][1]))
    output_ddho = pd.DataFrame(save_perturbed_ddho)
    output_ddho.columns = range(output_ddho.shape[1])
    output_ddho.to_csv(main_dir / 'task_5' / 'output_ddho_lisa_redone_v2.csv')

    output_change = pd.DataFrame(save_change_vector)
    output_change.columns = range(output_change.shape[1])
    output_change.to_csv(main_dir / 'task_5' / 'output_change_value_ddho_lisa_redone_v2.csv')

    # output_df['relative_ofv_change'] = output_df['new_ofv'] / last_objective_function
    output_df.to_csv(main_dir / 'task_5' / 'input_changes_lisa_redone_v2.csv')

    # print(popt)
