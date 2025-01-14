'''
created on 14.01.25
@author lisa
plots for task 4
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

main_dir = Path(r'C:\Users\lihel\Documents\MMUQ_Python_Setup\MMUQ_Python_Setup\EclipsePortable426\Data\mmuq_ws2425\hmg\UncertaintyQuantificationHydrology')

# read input data
df_change_value_inputs = pd.read_csv(main_dir / "task_4" / "input_changes_cython.csv", sep=";")
inp_dfe = pd.read_csv(main_dir / 'data' / 'time_series__24163005' / 'time_series___24163005.csv', sep=';', index_col=0)
inp_dfe.index = pd.to_datetime(inp_dfe.index, format='%Y-%m-%d-%H')

#==============================================================================
# Create plots of disturbed time series
#==============================================================================
# extract data needed
tems = inp_dfe.loc[:, 'tavg__ref'].values
ppts = inp_dfe.loc[:, 'pptn__ref'].values
change_values = df_change_value_inputs.loc[:, 'change_value_inputs'].values

# create perturbed time series
tems_perturbed = np.zeros((len(change_values), len(tems)))
for i in np.arange(len(change_values)):
    tems_perturbed[i] = change_values[i] * tems
tems_perturbed_min = np.min(tems_perturbed, axis=0)
tems_perturbed_max = np.max(tems_perturbed, axis=0)

ppts_perturbed = np.zeros((len(change_values), len(ppts)))
for i in np.arange(len(change_values)):
    ppts_perturbed[i] = change_values[i] * ppts
ppts_perturbed_min = np.min(ppts_perturbed, axis=0)
ppts_perturbed_max = np.max(ppts_perturbed, axis=0)

# create plots of perturbed time series
fig, axs = plt.subplots(2, 1, figsize=(4, 8), dpi=120, sharex=True)
(axs_tem, axs_ppt) = axs

# set label of sub axes
axs_tem.set_ylabel('TEM\n[°C]')
axs_ppt.set_ylabel('PPT\n[mm]')

# plot original input data
axs_tem.plot(inp_dfe['tavg__ref'], alpha=1.0)
axs_ppt.plot(inp_dfe['pptn__ref'], alpha=1.0)

# plot shaded region of perturbed time series
axs_tem.fill_between(inp_dfe.index, tems_perturbed_min, tems_perturbed_max, color='gray', alpha=0.7)
axs_ppt.fill_between(inp_dfe.index, ppts_perturbed_min, ppts_perturbed_max, color='gray', alpha=0.7)

# Some other makeup.
for ax in axs: ax.grid()
axs[-1].set_xlabel('Time [hr]')
plt.xticks(rotation=45)
plt.suptitle('Toller Titel tbd')
plt.show()
# fig.savefig(main_dir / 'task_4' / 'plots' / 'perturbed_time_series.png', bbox_inches='tight')
plt.close(fig)

