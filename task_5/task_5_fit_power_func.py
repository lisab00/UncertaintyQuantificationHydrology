import os
from pathlib import Path
from scipy.interpolate import PPoly
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import csv

main_dir = Path(r'C:\Users\lihel\Documents\MMUQ_Python_Setup\MMUQ_Python_Setup\EclipsePortable426\Data\mmuq_ws2425\hmg\UncertaintyQuantificationHydrology')

# read input data
inp_dfe = pd.read_csv(main_dir / 'data' / 'time_series__24163005' / 'time_series___24163005.csv', sep=';', index_col=0)
inp_dfe.index = pd.to_datetime(inp_dfe.index, format='%Y-%m-%d-%H')

ddho = inp_dfe.loc[:, 'ddho__ref'].values
diso = inp_dfe.loc[:, 'diso__ref'].values

# ensure ddho is sorted (required for PPoly)
sorted_indices = np.argsort(ddho)
ddho = ddho[sorted_indices]
diso = diso[sorted_indices]

# define intervals for fitted curves: ddho in [0,430), [430,end)
i = (np.abs(ddho - 430)).argmin()
ddho_s1 = ddho[:i]
diso_s1 = diso[:i]
diso_s2 = diso[i:]
ddho_s2 = ddho[i:]

# fit curves for each interval (fix: both poly must have same degree rn)
coeffs_s1 = np.polyfit(ddho_s1, diso_s1, deg=4)
coeffs_s2 = np.polyfit(ddho_s2, diso_s2, deg=4)
coeffs = np.array([coeffs_s1, coeffs_s2]).T

# analyze fit of seperate curves
P1 = PPoly(np.array([coeffs_s1]).T, [ddho[0], ddho[i]])
P2 = PPoly(np.array([coeffs_s2]).T, [ddho[i], ddho[-1]])
plt.scatter(ddho, diso, label="Observed Data", color="blue", alpha=0.2, s=5)
plt.plot(ddho_s1, P1(ddho_s1), color="red")
plt.plot(ddho_s2, P2(ddho_s2), color="green")
plt.show()

# align the two curves in one PPoly object
breakpoints = [ddho[0] , ddho[i], ddho[-1]]
power_func = PPoly(coeffs, breakpoints)

# compute discharge predicted (disp)
disp = power_func(ddho)

# plot
plt.scatter(ddho, diso, label="Observed Data", color="blue", alpha=0.2, s=5)
plt.plot(ddho, disp, label="Fitted Power Equation", color="red")
plt.xlabel("Depth (h)")
plt.ylabel("Discharge (Q)")
plt.legend()
plt.title("Fitted Power Equation to Depth-Discharge Data")
plt.show()
