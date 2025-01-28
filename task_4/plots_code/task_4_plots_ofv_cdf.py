'''
created on 26.01.25
@author lisa
plots for task 4
'''

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

main_dir = Path(r'C:\Users\lihel\Documents\MMUQ_Python_Setup\MMUQ_Python_Setup\EclipsePortable426\Data\mmuq_ws2425\hmg\UncertaintyQuantificationHydrology')

# import true ofv from task 1
iterations_df = pd.read_csv(main_dir / "task_1" / "outputs_task1" / "csv_outputs" / "output_one_per_iteration_tol_0.01_seed_123.csv")
iterations_df.columns = ["Obj_fct_value", "prm_value"]
ofv_true = iterations_df["Obj_fct_value"].iloc[-1]
print(ofv_true)

# read input data from task 4
df_inp_ch = pd.read_csv(main_dir / "task_4" / "output" / "input_changes.csv", index_col=0)
print(df_inp_ch["new_opt_ofv"])

sns.kdeplot(df_inp_ch["new_opt_ofv"], cumulative=True, color="blue", label="recalibrated")
plt.axvline(ofv_true, color="orange", linestyle="--", label="reference OFV")
plt.title("Reference OFV and recalibrated OFVs")
plt.xlabel("OFVs")
plt.ylabel("Cumulative distribution function")
plt.legend()
plt.show()

