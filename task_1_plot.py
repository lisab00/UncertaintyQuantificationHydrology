'''
Created on 06.11.2024

@author: hfran
'''
import pandas as pd
import matplotlib.pyplot as plt

obj_fct_values = pd.read_csv("C:\\Users\\hfran\\Documents\\Uni\\Master\\hydrology\\MMUQ_Python_Setup\\EclipsePortable426\\Data\\mmuq_ws2425\\hmg\\data\\task_1\\output_obj.csv", header=None)

obj_fct_values.columns = ["Obj_fct_value", "prm_value"]

fig = plt.figure()

plt.plot(list(range(1, len(obj_fct_values["Obj_fct_value"]) + 1)), obj_fct_values["Obj_fct_value"])

plt.grid()
plt.legend()

plt.xticks(rotation=45)

plt.xlabel('Optimization iteration')
plt.ylabel('Objective functionvalue')

plt.title('Plot of optimization progress')

plt.show()
fig.savefig(f'C:\\Users\\hfran\\Documents\\Uni\\Master\\hydrology\\MMUQ_Python_Setup\\EclipsePortable426\\Data\\mmuq_ws2425\\hmg\\data\\diff_evo_nse_seed_123_iteration.png', bbox_inches='tight')

plt.close(fig)

obj_fct_values = pd.read_csv("C:\\Users\\hfran\\Documents\\Uni\\Master\\hydrology\\MMUQ_Python_Setup\\EclipsePortable426\\Data\\mmuq_ws2425\\hmg\\data\\task_1\\output.csv", header=None)

obj_fct_values.columns = ["Obj_fct_value", "prm_value"]
df = pd.DataFrame(obj_fct_values['prm_value'])
print(df.head())
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
print(len(prm_names))
print(print(df['prm_value'].iloc[0]))

df['prm_value'] = df['prm_value'].str.replace("[ ", "")
df['prm_value'] = df['prm_value'].str.replace("[", "")
df['prm_value'] = df['prm_value'].str.replace("]", "")
df['prm_value'] = df['prm_value'].str.split()
df['prm_value'] = [[float(p) for p in prm] for prm in df['prm_value']]

df[prm_names] = pd.DataFrame(df['prm_value'].tolist(), index=df.index)
print(df.info())
print(len(df[prm_names[0]]))

for i in range(len(prm_names)):

    prm = prm_names[i]
    fig = plt.figure()

    plt.scatter(df[prm], obj_fct_values["Obj_fct_value"], alpha=0.5, s=0.5)

    plt.grid()
    plt.legend()

    plt.xticks(rotation=45)

    plt.xlabel(f'Parameter value of {prm}')
    plt.ylabel('Log of Objective function value')
    plt.yscale('log')

    plt.title('Parameter values during optimization')

    fig.savefig(f'C:\\Users\\hfran\\Documents\\Uni\\Master\\hydrology\\MMUQ_Python_Setup\\EclipsePortable426\\Data\\mmuq_ws2425\\hmg\\data\\diff_evo_nse_prm_values_seed_123_{prm}_log.png', bbox_inches='tight')
    plt.close(fig)

