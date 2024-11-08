'''
Created on 28.10.2024

@author: hfran, lisa
'''
import os
import sys
import time
import timeit
import traceback as tb
from pathlib import Path
# ...

import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt

from scipy.optimize import differential_evolution

from hmg.models import hbv1d012a_py
from hmg.test import aa_run_model
from hmg import HBV1D012A

#=============================================================================
# data preparation

# load data
# Absolute path to the directory where the input data lies.
# main_dir = Path(r'C:\Users\hfran\Documents\Uni\Master\hydrology\MMUQ_Python_Setup\EclipsePortable426\Data\mmuq_ws2425\hmg\data')
main_dir = Path(r'/Users/agnes_dchn/PycharmProjects/UncertaintyQuantificationHydrology/data')
os.chdir(main_dir)

# Read input text time series as a pandas Dataframe object and
# cast the index to a datetime object.
inp_dfe = pd.read_csv(main_dir/'time_series___24163005.csv', sep=';', index_col=0)
inp_dfe.index = pd.to_datetime(inp_dfe.index, format='%Y-%m-%d-%H')

# Read the catchment area in meters squared. The first value is needed
# only.
cca_srs = pd.read_csv(main_dir/'area___24163005.csv', sep=';', index_col=0)
ccaa = cca_srs.values[0, 0]

tems = inp_dfe.loc[:, 'tavg__ref'].values  # Temperature.
ppts = inp_dfe.loc[:, 'pptn__ref'].values  # Preciptiation.
pets = inp_dfe.loc[:, 'petn__ref'].values  # PET.
diso = inp_dfe.loc[:, 'diso__ref'].values  # Observed discharge.

tsps = tems.shape[0]  # Number of time steps.

# Conversion constant for mm/hour to m3/s.
dslr = ccaa / (3600 * 1000)  # For daily res. multiply denominator with 24.

#=============================================================================
# define different metrics


def nse(sim, obs):
    '''Nash-Suttcliffe Efficiency'''
    # return 1-nse ne nse to make minimization work
    return 1 - (1 - (np.sum((obs - sim) ** 2) / np.sum((obs - np.mean(obs)) ** 2)))


# lnNSE log of NSE, log of inputs, no 0
def lognse(sim, obs):
    '''log of NSE'''
    return nse(sim, obs)
    # return nse(sim, obs) if nse(sim, obs) > 0 else raise ValueError('log not defined')


def pbias(sim, obs):
    return 100 * (np.sum((sim - obs) / np.sum(obs)))


def rmse(sim, obs):
    return np.sqrt(np.mean((sim - obs) ^ 2))


def sse(sim, obs):
    return np.sum((sim - obs) ^ 2)


#=============================================================================
# define objective function
def obj_fun(prms, modl_objt, metric: str, diso):

    # # read out metric string
    # Dictionary mapping string names to metric functions
    metrics = {
        "sse": sse,
        "rmse": rmse,
        "nse": nse,
        "lognse": lognse,
        "pbias": pbias
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

    return metric_fun(diss, diso)

#==============================================================================
# define function to store intermediate results
save_data = [["Obj_fct_values","a","v","c","d","e","f","g","h","j","k","u","i","o","p","ü","ä","q","w","e","r","t","x","v"]]
def callback(intermediate_result):
    save_data.append([intermediate_result.fun, *intermediate_result.x])


#==============================================================================
# produce plots


def plot_output(otps, diss, diso, obj_fct_values):
    # Show a figure of the observed vs. simulated river flow.
    fig = plt.figure()

    plt.plot(inp_dfe.index, diso, label='REF', alpha=0.75)
    plt.plot(inp_dfe.index, diss, label='SIM', alpha=0.75)

    plt.grid()
    plt.legend()

    plt.xticks(rotation=45)

    plt.xlabel('Time [hr]')
    plt.ylabel('Discharge\n[$m^3.s^{-1}$]')

    plt.title('Observed vs. Simulated River Flow')

    plt.show()
    plt.close(fig)
    #===========================================================================

    # Show a figure of some of the internally simulated variables of the model.
    # This also serves as a diagnostic tool to check whether what is simulated
    # makes sense or not.
    fig, axs = plt.subplots(9, 1, figsize=(4, 8), dpi=120, sharex=True)

    (axs_tem,
     axs_ppt,
     axs_snw,
     axs_sl0,
     axs_sl1,
     axs_etn,
     axs_rrr,
     axs_rnf,
     axs_bal) = axs
    #===========================================================================

    # Inputs.
    axs_tem.plot(inp_dfe['tavg__ref'], alpha=0.85)
    axs_tem.set_ylabel('TEM\n[°C]')

    axs_ppt.plot(inp_dfe['pptn__ref'], alpha=0.85)
    axs_ppt.set_ylabel('PPT\n[mm]')
    #===========================================================================

    # Snow depth.
    axs_snw.plot(inp_dfe.index, otps[:, otps_lbls['snw_dth']], alpha=0.85)
    axs_snw.set_ylabel('SNW\n[mm]')
    #===========================================================================

    # Mositure level in both soil layers.
    axs_sl0.plot(inp_dfe.index, otps[:, otps_lbls['sl0_mse']], alpha=0.85)
    axs_sl0.set_ylabel('SL0\n[mm]')

    axs_sl1.plot(inp_dfe.index, otps[:, otps_lbls['sl1_mse']], alpha=0.85)
    axs_sl1.set_ylabel('SL1\n[mm]')
    #===========================================================================

    # Potential and simulated evapotranspiration.
    axs_etn.plot(inp_dfe.index, inp_dfe['petn__ref'], label='PET', alpha=0.85)

    axs_etn.plot(
        inp_dfe.index, otps[:, otps_lbls['sl1_etn']], label='ETN', alpha=0.85)

    axs_etn.set_ylabel('ETN\n[mm]')
    axs_etn.legend()
    #===========================================================================

    # Depth of water in the upper and lower reservoirs.
    axs_rrr.plot(
        inp_dfe.index, otps[:, otps_lbls['urr_dth']], label='URR', alpha=0.85)

    axs_rrr.plot(
        inp_dfe.index, otps[:, otps_lbls['lrr_dth']], label='LRR', alpha=0.85)

    axs_rrr.set_ylabel('DTH\n[mm]')
    axs_rrr.legend()
    #===========================================================================

    # Surface and underground runoff.
    axs_rnf.plot(
        inp_dfe.index, otps[:, otps_lbls['rnf_sfc']], label='SFC', alpha=0.85)

    axs_rnf.plot(
        inp_dfe.index, otps[:, otps_lbls['rnf_gnd']], label='GND', alpha=0.85)

    axs_rnf.set_ylabel('RNF\n[mm]')
    axs_rnf.legend()
    #===========================================================================

    # Water balance time series at each time step.
    # Should be close to zero.
    axs_bal.plot(inp_dfe.index, otps[:, otps_lbls['mod_bal']], alpha=0.85)
    axs_bal.set_ylabel('BAL\n[mm]')
    #===========================================================================

    # Some other makeup.
    for ax in axs: ax.grid()

    axs[-1].set_xlabel('Time [hr]')

    plt.xticks(rotation=45)

    plt.suptitle('Inputs, and internally simulated variables of HBV')
    plt.show()

    plt.close(fig)

    #==========================================================================

    # Plot optimization curve
    fig = plt.figure()

    plt.plot(list(range(1, len(obj_fct_values) + 1)), obj_fct_values)

    plt.grid()
    plt.legend()

    plt.xticks(rotation=45)

    plt.xlabel('Optimization iteration')
    plt.ylabel('Objective functionvalue')

    plt.title('Plot of optimization progress')

    plt.show()
    plt.close(fig)

#==============================================================================
# main code


# # preparations enabling proper work with model (explanations see aa_run_model)
modl_objt = HBV1D012A()
modl_objt.set_inputs(tems, ppts, pets)
modl_objt.set_outputs(tsps)
modl_objt.set_discharge_scaler(dslr)
otps_lbls = modl_objt.get_output_labels()

# # perform task

# manually set prms bounds
# bounds copied from standard model (also see moodle)
bounds_dict = {
        'snw_dth': (0.00, 0.00),
        'snw_ast': (-1.0, +1.0),
        'snw_amt': (-0.0, +2.0),
        'snw_amf': (0.00, 2.00),
        'snw_pmf': (0.00, 2.00),

        'sl0_mse': (0.00, 1e+2),
        'sl1_mse': (0.00, 2e+2),

        'sl0_fcy': (0.00, 2e+2),
        'sl0_bt0': (0.00, 3.00),

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



# set metric that should be used
metric = "nse"

# # implement optimization using differential evolution algo
# bevor each run make sure to update the path to csv file for intermediate results
# in the callback function

res = differential_evolution(func=obj_fun,  # function to be minimized
                             args=(modl_objt, metric, diso),  # fixed args for func
                             bounds=list(bounds_dict.values()),  # bounds on prms
                            # maxiter=1,  # max number of iterations to be performed
                             callback=callback,  # write intermediate values to csv file
                             tol=0.01,  # stopping criterion
                             seed=10,  # make stochastic minimization reproducible
                             disp=True,  # print intermediate results
                             polish=False)  # always set this to false


with open(main_dir / "output.csv", "w", newline="") as csvfile: # main_dir.read_text()
    writer = csv.writer(csvfile, delimiter=';')
    writer.writerows(save_data)
# obtain fitted prms
res_prms = res.x
res_suc = res.success
res_fun_val = 1 - res.fun

# print outputs
print(f"optimized prms: {res_prms}")
print(f"success: {res_suc}")
print(f"value of performance metric: {res_fun_val}")
print(f"message: {res.message}")
print(f"number of iterations performed: {res.nit}")

# produce plots
otps = modl_objt.get_outputs()
diss = modl_objt.get_discharge()
# obj_fct_values = pd.read_csv("C:\\Users\\hfran\\Documents\\Uni\\Master\\hydrology\\MMUQ_Python_Setup\\EclipsePortable426\\Data\\mmuq_ws2425\\hmg\\data\\task_1\\output.csv", header=None)
interm_res = pd.read_csv("/Users/agnes_dchn/PycharmProjects/UncertaintyQuantificationHydrology/data/output.csv", delimiter=';')

plot_output(otps, diss, diso, interm_res['Obj_fct_values'])
#####
output_path = "/Users/agnes_dchn/PycharmProjects/UncertaintyQuantificationHydrology/data/simulated_discharge.csv"
simulated_discharge_df = pd.DataFrame({"Time": inp_dfe.index, "Simulated_Discharge": diss})
simulated_discharge_df.to_csv(output_path, sep=";", index=False)

print(f"Simulated discharge data exported to: {output_path}")