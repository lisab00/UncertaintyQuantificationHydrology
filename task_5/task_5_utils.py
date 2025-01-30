import os
from pathlib import Path
from scipy.interpolate import PPoly
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import csv


def compute_disp(ddho, coeffs, i):
    '''evaluate power function on ddho and return predicted discharge disp
    input should be perturbed ddho
    '''
    disp_s1 = np.polyval(coeffs[0], ddho[:i + 1])
    disp_s2 = np.polyval(coeffs[1], ddho[i:])
    disp_s2_concate = np.delete(disp_s2, 0)
    disp = np.concatenate((disp_s1, disp_s2_concate))
    return disp


def compute_coeffs(ddho, diso, i, degree, num_outliers=55):
    '''returns coeffs for rating curve
    inputs are original ddho / diso time series
    put i as: i = (np.abs(ddho - 430)).argmin()
    '''

    # define intervals for fitted curves
    ddho_s1 = ddho[:i + 1]
    diso_s1 = diso[:i + 1]
    diso_s2 = diso[i:]
    ddho_s2 = ddho[i:]

    # fit curves for each interval
    coeffs_s1 = np.polyfit(ddho_s1, diso_s1, deg=degree)
    coeffs_s2 = np.polyfit(ddho_s2, diso_s2, deg=degree)

    return (coeffs_s1, coeffs_s2)

# ============================================================================
# run this once


main_dir = Path(r'C:\Users\lihel\Documents\MMUQ_Python_Setup\MMUQ_Python_Setup\EclipsePortable426\Data\mmuq_ws2425\hmg\UncertaintyQuantificationHydrology')

inp_dfe = pd.read_csv(main_dir / 'data' / 'time_series__24163005' / 'time_series___24163005.csv', sep=';', index_col=0)
inp_dfe.index = pd.to_datetime(inp_dfe.index, format='%Y-%m-%d-%H')

ddho = inp_dfe.loc[:, 'ddho__ref'].values
diso = inp_dfe.loc[:, 'diso__ref'].values

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

# decide on degree and index for power function. do this once to obtain coeffs
degree = 12
i = (np.abs(ddho - 430)).argmin()
coeffs = compute_coeffs(ddho, diso, i, degree)

# ============================================================================
# run this every loop
# call compute_disp for evers perturbed time series with same coeffs
# make sure you perturb time series where the outliers are removed!
disp = compute_disp(ddho, coeffs, i)

# ============================================================================
# plot as sanity check that my code works
'''plt.plot(ddho, disp, color="red")
plt.scatter(ddho, diso, label="Observed Data", color="blue", alpha=0.2, s=5)
plt.xlabel("Depth (h)")
plt.ylabel("Discharge (Q)")
plt.legend()
plt.title("Fitted Power Equation to Depth-Discharge Data")
plt.show()'''

