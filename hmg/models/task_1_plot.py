'''
Created on 06.11.2024

@author: hfran
'''
import pandas as pd
import matplotlib.pyplot as plt

obj_fct_values = pd.read_csv('/Users/agnes_dchn/Documents/Uni/1_Master/2_Semester/MAth and opti/Opim_prozesse_ausblenden/output_lrr_ausgeblendet.csv', header=None)
print(f"Erste 5 Zeilen von {obj_fct_values}:")

#obj_fct_values.columns = ["Obj_fct_value", "prm_value"]
#df = pd.read_csv(obj_fct_values, sep=',', header=None)

#  manuell Spaltennamen zuweisen
#df.columns = ["Obj_fct_value", "prm_value"]
obj_fct_values.columns = ["Obj_fct_value","prm_value"]

# Überprüfen Daten
print(f"Erste 5 Zeilen von {obj_fct_values}:")


print(obj_fct_values.head())

fig = plt.figure()

plt.plot(list(range(1, len(obj_fct_values["Obj_fct_value"]) + 1)), obj_fct_values["Obj_fct_value"])

plt.grid()
plt.legend()

plt.xticks(rotation=45)

plt.xlabel('Optimization iteration')
plt.ylabel('Objective functionvalue')

plt.title('Plot of optimization progress')

plt.show()
plt.close(fig)

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

fig = plt.figure()

plt.scatter(df['snw_amt'], obj_fct_values["Obj_fct_value"])

plt.grid()
plt.legend()

plt.xticks(rotation=45)

plt.xlabel('Parameter value of snw_amt')
plt.ylabel('Objective function value')

plt.title('Plot of optimization progress')

plt.show()
plt.close(fig)
