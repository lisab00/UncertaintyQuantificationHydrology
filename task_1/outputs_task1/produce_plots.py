'''
Created on 11.11.2024

@author: lisa
'''

import pandas as pd
import matplotlib.pyplot as plt


def plot_fitted_curve(diss, diso, index):
    # Show a figure of the observed vs. simulated river flow.
    fig = plt.figure()

    plt.plot(index, diso, label='REF', alpha=0.75)
    plt.plot(index, diss, label='SIM', alpha=0.75)

    plt.grid()
    plt.legend()

    plt.xticks(rotation=45)

    plt.xlabel('Time [hr]')
    plt.ylabel('Discharge\n[$m^3.s^{-1}$]')

    plt.title('Observed vs. Simulated River Flow')

    plt.show()
    plt.close(fig)

def plot_optim_curve(obj_fct_values):
    # Plot of the optimization progress
    fig = plt.figure()

    plt.plot(list(range(1, len(obj_fct_values) + 1)), obj_fct_values)

    plt.grid()
    plt.legend()

    plt.xticks(rotation=45)

    plt.xlabel('Optimization iteration')
    plt.ylabel('Objective functionvalue')

    plt.title('Plot of optimization progress')

    plt.show()
    plt.close(fig)


def scatter_plot_parm(output, prm_name: str):
    # scatter plot the parameters against the objective function values
    # during the optimization progress

    fig = plt.figure()

    plt.scatter(output[prm_name], output['Obj_fct_values'])

    plt.grid()
    plt.legend()

    plt.xticks(rotation=45)

    plt.xlabel(f'Parameter value of {prm_name}')
    plt.ylabel('Objective function value')

    plt.title('Scatter plot of parameter vs. objective function value')

    plt.show()
    plt.close(fig)
