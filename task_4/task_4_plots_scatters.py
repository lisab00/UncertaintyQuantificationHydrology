'''
created on 28.01.25
@author lisa
plots for task 4
'''

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

main_dir = Path(r'C:\Users\lihel\Documents\MMUQ_Python_Setup\MMUQ_Python_Setup\EclipsePortable426\Data\mmuq_ws2425\hmg\UncertaintyQuantificationHydrology')

# read input data from task 4
df_inp_ch = pd.read_csv(main_dir / "task_4" / "output" / "input_changes.csv", index_col=0)
ofv_old = 1 - df_inp_ch["new_ofv_old_params"]
ofv_recal = 1 - df_inp_ch["new_opt_ofv"]

fig = plt.figure()
plt.scatter(ofv_old, ofv_recal)
plt.title("Compare perturbed against recalibrated NSE")
plt.xlabel("Perturbed NSE before recalibration")
plt.ylabel("Recalibrated NSE")
plt.show()
fig.savefig(main_dir / 'task_4' / 'plots' / 'ofv_scatter_old_recal.png', bbox_inches='tight')
plt.close(fig)

