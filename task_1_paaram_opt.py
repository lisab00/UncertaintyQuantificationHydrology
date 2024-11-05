'''
Created on 28.10.2024

@author: hfran
'''
import os
import sys
import time
import timeit
import traceback as tb
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from hmg import HBV1D012A
from models import hbv1d012a_py
from test import aa_run_model

# use scipy for optimization
# from sklearn.model_selection import GridSearchCV

#=============================================================================
# load data and params

# Absolute path to the directory where the input data lies.
main_dir = Path(r'C:\Users\hfran\Documents\Uni\Master\hydrology\MMUQ_Python_Setup\EclipsePortable426\Data\mmuq_ws2425\hmg\data')
os.chdir(main_dir)

# Read input text time series as a pandas Dataframe object and
# cast the index to a datetime object.
inp_dfe = pd.read_csv(r'time_series___24163005.csv', sep=';', index_col=0)
inp_dfe.index = pd.to_datetime(inp_dfe.index, format='%Y-%m-%d-%H')

# Read the catcment area in meters squared. The first value is needed
# only.
cca_srs = pd.read_csv(r'area___24163005.csv', sep=';', index_col=0)
ccaa = cca_srs.values[0, 0]

tems = inp_dfe.loc[:, 'tavg__ref'].values  # Temperature.
ppts = inp_dfe.loc[:, 'pptn__ref'].values  # Preciptiation.
pets = inp_dfe.loc[:, 'petn__ref'].values  # PET.
diso = inp_dfe.loc[:, 'diso__ref'].values  # Observed discharge.

tsps = tems.shape[0]  # Number of time steps.

# Conversion constant for mm/hour to m3/s.
dslr = ccaa / (3600 * 1000)  # For daily res. multiply denominator with 24.

# Read model and related files in the models directory for more info.
# Correct sequence must be followed. Values that are out of
# the absolute parameter range will result in an AssertionError.

# turn off for all processes

prms = np.array([
    0.00,  # 'snw_dth'
    +0.1,  # 'snw_ast'
    -0.1,  # 'snw_amt'
    0.01,  # 'snw_amf'
    0.00,  # 'snw_pmf'

    50.0,  # 'sl0_mse'
    300.,  # 'sl1_mse'

    70.0,  # 'sl0_fcy'
    2.50,  # 'sl0_bt0'

    300.,  # 'sl1_pwp'
    400.,  # 'sl1_fcy'
    2.50,  # 'sl1_bt0'

    0.00,  # 'urr_dth'
    0.00,  # 'lrr_dth'

    1.00,  # 'urr_rsr'
    30.0,  # 'urr_tdh'
    0.15,  # 'urr_tdr'
    1e-4,  # 'urr_cst'
    1.00,  # 'urr_dro'
    0.00,  # 'urr_ulc'

    0.00,  # 'lrr_tdh'
    0.00,  # 'lrr_cst'
    0.00,  # 'lrr_dro'
    ], dtype=np.float32)

PINF = +np.float32(np.inf)
NINF = -np.float32(np.inf)

buds = {
        'snw_dth': (0.00, PINF),
        'snw_ast': (NINF, PINF),
        'snw_amt': (NINF, PINF),
        'snw_amf': (0.00, PINF),
        'snw_pmf': (0.00, PINF),

        'sl0_mse': (0.00, PINF),
        'sl1_mse': (0.00, PINF),

        'sl0_fcy': (0.00, PINF),
        'sl0_bt0': (0.00, PINF),

        'sl1_pwp': (0.00, PINF),
        'sl1_fcy': (0.00, PINF),
        'sl1_bt0': (0.00, PINF),

        'urr_dth': (0.00, PINF),
        'lrr_dth': (0.00, PINF),

        'urr_rsr': (0.00, 1.00),
        'urr_tdh': (0.00, PINF),
        'urr_tdr': (0.00, 1.00),
        'urr_cst': (0.00, 1.00),
        'urr_dro': (0.00, 1.00),
        'urr_ulc': (0.00, 1.00),

        'lrr_tdh': (0.00, PINF),
        'lrr_cst': (0.00, 1.00),
        'lrr_dro': (0.00, 1.00),
    }

#=============================================================================
# metrics


def nse(sim, obs):
    '''Nash-Suttcliffe Efficiency'''
    return -1 * (1 - (np.sum((obs - sim) ** 2) / np.sum((obs - np.mean(obs)) ** 2)))
# lnNSE log of NSE, log of inputs, no 0


def pbias(sim, obs):
    return 100 * (np.sum((sim - obs) / np.sum(obs)))


def rmse(sim, obs):
    return np.sqrt(np.mean((sim - obs) ** 2))


def sse(sim, obs):
    return np.sum((sim - obs) ** 2)


#==========================================================================
def obj_function(prms, obs, accuracy, buds_dict):

    # obj_fct_value = sse(sim, obs)

    # Initiate an empty model object. To understand what each method call
    # used below is for, please take a look at the files inside models
    # directory.
    modl_objt = HBV1D012A()

    # Set the above defined inputs.
    modl_objt.set_inputs(tems, ppts, pets)

    # Pass the number of time steps to the model object here. It creates the
    # ouputs array(s) with the proper shape.
    modl_objt.set_outputs(tsps)

    # Set the constant that will convert units from those of precipitation
    # to those of measured discharge.
    modl_objt.set_discharge_scaler(dslr)
    #==========================================================================
    # Get a dictionary that links an output labe to its column index in the
    # ouputs array.
    otps_lbls = modl_objt.get_output_labels()

    # Pass the parameters.
    modl_objt.set_parameters(prms)

    bounds = modl_objt.get_parameter_bounds_in_correct_order(buds_dict)

    # Tell the model object that the simulation is a not an optimization.
    modl_objt.set_optimization_flag(0)

    # Run the model for the given inputs, constants and parameters.
    modl_objt.run_model()

    # Read the internal ouputs and simulated discharge.
    otps = modl_objt.get_outputs()
    diss = modl_objt.get_discharge()

    # sim = model(params)
    # compute rmse(sim, obs) =ofv?
    # Q: what is obj fct value?

    return diss, otps, otps_lbls, bounds


def objective_function(prms, accuracy):

    # Q: What is least number of processes required?
    # bedeutet: einzelne Parameter = 0 setzen
    diss, otps, otps_lbls, bounds = obj_function(prms, diso, accuracy, buds)
    obj_fct_value = accuracy(diss, diso)
    return obj_fct_value
#==============================================================================
# produce plots


def plot_output(diss, otps, otps_lbls, prm, conclusion):  # TODO define inputs
    # Show a figure of the observed vs. simulated river flow.
    fig = plt.figure()

    plt.plot(inp_dfe.index, diso, label='REF', alpha=0.75)
    plt.plot(inp_dfe.index, diss, label='SIM', alpha=0.75)

    plt.grid()
    plt.legend()

    plt.xticks(rotation=45)

    plt.xlabel('Time [hr]')
    plt.ylabel('Discharge\n[$m^3.s^{-1}$]')

    plt.title('Observed vs. Simulated RIver Flow')

    # plt.show()
    fig.savefig(f'C:\\Users\\hfran\\Documents\\Uni\\Master\\hydrology\\MMUQ_Python_Setup\\EclipsePortable426\\Data\\mmuq_ws2425\\hmg\\data\\lbfgs_nse_{conclusion}.png')
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
    #==========================================================================

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
    # plt.show()
    fig.savefig(f'C:\\Users\\hfran\\Documents\\Uni\\Master\\hydrology\\MMUQ_Python_Setup\\EclipsePortable426\\Data\\mmuq_ws2425\\hmg\\data\\lbfgs_nse_2.png')

    plt.close(fig)


counter = 1
obj_fct_values = {}
prm_values = {}


def callback(x):
    diss_2, otps_2, otps_lbls_2, bounds_prms = obj_function(x, diso, pbias, buds)
    error = nse(diso, diss_2)
    global counter
    counter += 1
    global obj_fct_values
    obj_fct_values[counter] = error
    global prm_values
    prm_values[counter] = x
    # print(obj_fct_values)


if __name__ == '__main__':
    # obj_fct_value, diss, otps, otps_lbls = obj_function(diso, nse)
    # print(obj_fct_value)
    diss_2, otps_2, otps_lbls_2, bounds_prms = obj_function(prms, diso, nse, buds)

    prm_names = list(buds.keys())

    print(bounds_prms)
    ''' for i in range(len(prms)):
        prms[i] = 0
        bounds_prms[i] = [0, 0]          # nummer zuweisungen immer an anfang schreiben
        res = minimize(
                objective_function,
                method='SLSQP',
                x0=prms,
                args=(pbias),
                options={'maxiter':50, 'disp':True},
                bounds=bounds_prms,
                )
        print(res)
        x_param = res.x
        x_success = res.success
        x_message = res.message
        x_fun_value = res.fun
        conclusion = f"terminated{x_success}_with_obj_value{x_fun_value}"
        print(x_param)
        diss_3, otps_3, otps_lbls_3, bounds_prms = obj_function(x_param, diso, rmse, buds)

        plot_output(diss_3, otps_3, otps_lbls_3, prm_names[i], conclusion)'''

    '''for i in range(len(prms)):
        for j in range(len(prms)):
            if i != j:

                given_prm_value = prms[j]
                bounds_prms[j] = [given_prm_value, given_prm_value]

        res = minimize(
                objective_function,
                method='L-BFGS-B',
                x0=prms,
                args=(rmse),
                options={'maxiter':50, 'disp':True},
                bounds=bounds_prms,
                callback=callback,
                )
        print(res)
        x_param = res.x
        x_success = res.success
        x_message = res.message
        x_fun_value = res.fun
        conclusion = f"terminated{x_success}_with_obj_value{x_fun_value}"
        print(x_param)
        diss_3, otps_3, otps_lbls_3, bounds_prms = obj_function(x_param, diso, rmse, buds)
        print(obj_fct_values)
        plot_output(diss_3, otps_3, otps_lbls_3, prm_names[i], conclusion)

'''
    res = minimize(
                objective_function,
                method='L-BFGS-B',
                x0=prms,
                args=(nse),
                options={'maxiter':50, 'disp':True},
                bounds=bounds_prms,
                callback=callback,
                )
    print(res)
    x_param = res.x
    x_success = res.success
    x_message = res.message
    x_fun_value = res.fun
    conclusion = f"terminated{x_success}_with_obj_value{x_fun_value}"
    print(x_param)
    diss_3, otps_3, otps_lbls_3, bounds_prms = obj_function(x_param, diso, nse, buds)
    print(obj_fct_values)
    plot_output(diss_3, otps_3, otps_lbls_3, prm_names, conclusion)

# snow melt, soil, urr, lrr
# plot each parameter value against its obj function values
# plot obj fct value against iteration
