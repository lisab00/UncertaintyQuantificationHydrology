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

# read true time series
df_ppts = pd.read_csv(main_dir / "data" / "time_series__24163005" / "time_series___24163005.csv", sep=';')

# read input data from task 4
df_pert_ppts = pd.read_csv(main_dir / "task_4" / "output" / "output_ppts.csv", index_col=0)
df_pert_tems = pd.read_csv(main_dir / "task_4" / "output" / "output_tems.csv", index_col=0)
# df_pert_tems = pd.read_csv(main_dir / "task_4" / "test_csv" / "output_tems.csv", index_col=0)

# plot for ppts
'''for i in np.arange(len(df_pert_ppts)):
    if i == 0:
        sns.kdeplot(df_pert_ppts.iloc[i], cumulative=True, alpha=0.5, color='orange', label='perturbed', linewidth=3)
    else:
        sns.kdeplot(df_pert_ppts.iloc[i], cumulative=True, alpha=0.5, color='orange', linewidth=3)

    plt.title("Compare original and perturbed ppts time series")
    plt.xlabel("ppts values")
    plt.ylabel("Cumulative distribution function")
    if i % 100 == 0:
        print(f"{i} plots created")

sns.kdeplot(df_ppts["pptn__ref"], cumulative=True, color="blue", label="original", linewidth=1)
plt.legend()
plt.show()'''

# plot for tems
for i in np.arange(len(df_pert_tems)):
    if i == 0:
        sns.kdeplot(df_pert_tems.iloc[i], cumulative=True, alpha=0.5, color='orange', label='perturbed', linewidth=3)
    else:
        sns.kdeplot(df_pert_tems.iloc[i], cumulative=True, alpha=0.5, color='orange', linewidth=3)

    plt.title("Compare original and perturbed temperature time series")
    plt.xlabel("Temperature values")
    plt.ylabel("Cumulative distribution function")
    if i % 100 == 0:
        print(f"{i} plots created")

sns.kdeplot(df_ppts["tavg__ref"], cumulative=True, color="blue", label="original", linewidth=1)
plt.legend()
plt.show()
