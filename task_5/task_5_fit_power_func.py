import os
from pathlib import Path
from scipy.interpolate import PPoly
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import csv

main_dir = Path(r'C:\Users\lihel\Documents\MMUQ_Python_Setup\MMUQ_Python_Setup\EclipsePortable426\Data\mmuq_ws2425\hmg\UncertaintyQuantificationHydrology')


def abs_error(diss, diso, error_bound):
    """returns true, if the maximum of abs(diss-diso) <= 1mm
    returns false if stopping criterion is not reached
    """
    abs_dist = np.abs(diss - diso)
    print(f" max abs_dist = {max(abs_dist)}")
    return np.all(abs_dist <= error_bound)


if __name__ == "__main__":

    # read input data
    inp_dfe = pd.read_csv(main_dir / 'data' / 'time_series__24163005' / 'time_series___24163005.csv', sep=';', index_col=0)
    inp_dfe.index = pd.to_datetime(inp_dfe.index, format='%Y-%m-%d-%H')

    ddho = inp_dfe.loc[:, 'ddho__ref'].values
    diso = inp_dfe.loc[:, 'diso__ref'].values

    plt.scatter(ddho, diso, label="Observed Data", color="blue", alpha=0.2, s=5)
    plt.show()

    # ensure ddho is sorted (required for PPoly)
    sorted_indices = np.argsort(ddho)
    ddho = ddho[sorted_indices]
    diso = diso[sorted_indices]

    #=============================================================================
    # fit one polynomial to whole time series
    coeffs_complete_fit = np.polyfit(ddho, diso, deg=12)

    # residual analysis to eliminate outliers
    residuals = abs(diso - np.polyval(coeffs_complete_fit, ddho))
    print(f"res not sorted: {residuals}")

    res_sorted_indices = np.argsort(-residuals)
    residuals = residuals[res_sorted_indices]
    print(f"res sorted: {residuals}")

    num_outliers = 55
    outlier_indices = res_sorted_indices[:num_outliers]
    print(f"indices of outliers: {outlier_indices}")

    ddho = np.delete(ddho, outlier_indices)
    diso = np.delete(diso, outlier_indices)
    print(f"len ddho: {len(ddho)}")

    plt.plot(ddho, np.polyval(coeffs_complete_fit, ddho), color="red")
    plt.scatter(ddho, diso, label="Observed Data", color="blue", alpha=0.2, s=5)
    plt.show()

    # define intervals for fitted curves: ddho in [0,430), [430,end)
    i = (np.abs(ddho - 430)).argmin()
    ddho_s1 = ddho[:i + 1]
    diso_s1 = diso[:i + 1]
    diso_s2 = diso[i:]
    ddho_s2 = ddho[i:]
    print(f"ddho_s1: {ddho_s1[i]}")
    print(f"ddho_s2: {ddho_s2[0]}")
    print(ddho_s1[-1] == ddho_s2[0])
    print(f"segment two starts at index {i}\n")

    #=============================================================================
    # fit curves for each interval (fix: both poly must have same degree rn)
    degree = 20  # decide on degree here
    coeffs_s1 = np.polyfit(ddho_s1, diso_s1, deg=degree)
    print(f"c1: {coeffs_s1}, shape: {coeffs_s1.shape}")
    coeffs_s2 = np.polyfit(ddho_s2, diso_s2, deg=degree)
    print(f"c2: {coeffs_s2}, shape: {coeffs_s2.shape}")
    coeffs_piecewise = np.vstack((coeffs_s1, coeffs_s2)).T
    print(f"coeffs of piecewise poly: {coeffs_piecewise}, shape: {coeffs_piecewise.shape}")

    # discharge predicted (disp)
    # power function is evaluated here. Must use coeffs_s1 ond ddho_s1 series
    # and coeffs_s2 on ddho_s2 series
    disp_s1 = np.polyval(coeffs_s1, ddho_s1)
    disp_s2 = np.polyval(coeffs_s2, ddho_s2)
    disp_s2_concate = np.delete(disp_s2, 0)
    disp = np.concatenate((disp_s1, disp_s2_concate))

    # analyze fit of seperate curves
    plt.plot(ddho_s1, disp_s1, color="red", label="Fitted segment 1")
    plt.plot(ddho_s2, disp_s2, color="orange", label="Fitted segment 2")
    plt.scatter(ddho, diso, label="Observed Data", color="blue", alpha=0.2, s=5)
    plt.xlabel("Depth (h)")
    plt.ylabel("Discharge (Q)")
    plt.legend()
    plt.title("Fitted Power Equation to Depth-Discharge Data")
    plt.show()

    # measure error
    error_bound = 4.1
    print(f"error is small enough: {abs_error(disp, diso, error_bound)}")

    # ========================================================================
    # fancy way of creating piecewise polynomial using PPoly package but the
    # gods don't want that it works for me

    '''# align the two curves in one PPoly object
    breakpoints = np.array([ddho_s1[0], ddho_s1[-1], ddho_s2[-1]])
    print(breakpoints.shape)
    power_func = PPoly(coeffs_piecewise, breakpoints)

    # compute discharge predicted (disp)
    disp = power_func(ddho)

    # plot
    plt.scatter(ddho, diso, label="Observed Data", color="blue", alpha=0.2, s=5)
    plt.plot(ddho, disp, label="Fitted Power Equation", color="red")
    plt.xlabel("Depth (h)")
    plt.ylabel("Discharge (Q)")
    plt.legend()
    plt.title("Fitted Power Equation to Depth-Discharge Data")
    plt.show()'''

