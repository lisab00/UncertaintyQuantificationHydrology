'''
created on 14.01.25
@author lisa
plots for task 4
'''

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_perturbed_time_series():
    '''
    Create plots of disturbed time series
    '''

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
    # plt.show()
    fig.savefig(main_dir / 'task_4' / 'plots' / 'perturbed_time_series.png', bbox_inches='tight')
    plt.close(fig)


def plot_ofv_vs_change(ofvs, opt: bool):
    '''
    Create plots of ofvs vs change values
    '''

    plt.scatter(change_values, ofvs)
    plt.axhline(y=0.908, color='orange', linestyle='--', linewidth=2.5, label='Optimal OFV of original time series')
    plt.legend()
    plt.xlabel('Factor of parameter perturbation')

    if opt == True:
        plt.title('OFVs vs. perturbation factor')
        plt.ylabel('Objective function value')
    else:
        plt.title('Recalibrated OFVs vs. perturbation factor')
        plt.ylabel('Recalibrated objective function value')

    if opt == True:
        plt.savefig(main_dir / 'task_4' / 'plots' / 'ofv_perturbed_prms.png', bbox_inches='tight')
    else:
        plt.savefig(main_dir / 'task_4' / 'plots' / 'recalibrated_ofv_perturbed_prms.png', bbox_inches='tight')

    # plt.show()
    plt.close()


def box_plot_all(new_opt_prms, prm_names, flag=False):

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.boxplot(new_opt_prms, positions=np.arange(1, len(prm_names) + 1))

    ax.set_xlabel("Parameters")
    ax.set_ylabel("Parameter values")

    ax.set_xticks(np.arange(1, len(prm_names) + 1))
    ax.set_xticklabels(prm_names, rotation=45, ha='right')
    # ax.set_yscale('log')

    if flag == True:
        ax.set_title("Boxplots of recalibrated parameter values without \"lrr_tdh\"")
        fig.savefig(main_dir / 'task_4' / 'plots' / 'boxplots' / 'all_boxplots_no_lrr_tdh.png', bbox_inches='tight')

    else:
        ax.set_title("Boxplots of recalibrated parameter values")
        fig.savefig(main_dir / 'task_4' / 'plots' / 'boxplots' / 'all_boxplots.png', bbox_inches='tight')

    # plt.show()
    plt.close(fig)


def box_plot_prm(prm_name: str, prm_values, prm_optimized: float):

    box = plt.boxplot(prm_values)
    median_line = box["medians"][0]

    plt.xlabel(f"{prm_name}")
    plt.ylabel("Recalibrated values")
    plt.title(f"Boxplot of recalibrated \"{prm_name}\" values")
    plt.axhline(y=prm_optimized, color='blue', linestyle='--', linewidth=1.5)
    plt.legend(handles=[plt.Line2D([0], [0], color='blue', linestyle='--', linewidth=1.5),
                   median_line],
           labels=[f"Original parameter: {prm_optimized}", f"Median: {median_line.get_ydata()[0]:.3f}"], loc="best")

    plt.xticks([])
    # plt.yticks(np.linspace(min(prm_values), max(prm_values), 4))
    plt.yticks([])

    plt.savefig(main_dir / 'task_4' / 'plots' / 'boxplots' / f'{prm_name}_boxplot.png', bbox_inches='tight')
    # plt.show()
    plt.close()


def violin_plot_all(new_opt_prms, prm_names, flag=False):

    fig, ax = plt.subplots(figsize=(12, 6))

    sns.violinplot(data=new_opt_prms)

    ax.set_xlabel("Parameters")
    ax.set_ylabel("Parameter values")

    ax.set_xticks(np.arange(len(prm_names)))
    ax.set_xticklabels(prm_names, rotation=45, ha='right')

    if flag == True:
        fig.savefig(main_dir / 'task_4' / 'plots' / 'violinplots' / f'all_violinplots_no_lrr_tdh.png', bbox_inches='tight')
        ax.set_title("Violin plots of recalibrated parameter values without \"lrr_tdh\"")

    else:
        ax.set_title("Violin plots of recalibrated parameter values")
        fig.savefig(main_dir / 'task_4' / 'plots' / 'violinplots' / f'all_violinplots.png', bbox_inches='tight')

    # plt.show()
    plt.close(fig)


def violin_plot_prm(prm_name: str, prm_values, prm_optimized: float):

    sns.violinplot(data=prm_values)

    plt.xlabel(f"{prm_name}")
    plt.ylabel("Recalibrated values")
    # plt.axhline(y=prm_optimized, color='blue', linestyle='--', linewidth=1.5, label='original parameter')
    plt.title(f"Violin plot of recalibrated \"{prm_name}\" values")
    plt.legend([f"original parameter: {prm_optimized}"])

    plt.savefig(main_dir / 'task_4' / 'plots' / 'violinplots' / f'{prm_name}_violinplot.png', bbox_inches='tight')
    # plt.show()
    plt.close()


if __name__ == "__main__":

    main_dir = Path(r'C:\Users\lihel\Documents\MMUQ_Python_Setup\MMUQ_Python_Setup\EclipsePortable426\Data\mmuq_ws2425\hmg\UncertaintyQuantificationHydrology')

    # read input data from current task
    df_change_value_inputs = pd.read_csv(main_dir / "task_4" / "input_changes_cython_2000.csv", sep=";")

    print(df_change_value_inputs.head())

    # read input data from old tasks
    inp_dfe = pd.read_csv(main_dir / 'data' / 'time_series__24163005' / 'time_series___24163005.csv', sep=';', index_col=0)
    inp_dfe.index = pd.to_datetime(inp_dfe.index, format='%Y-%m-%d-%H')
    df_cumulated_output = pd.read_csv(main_dir / 'task_2' / 'cumulated_output_0.05.csv' , sep=';')
    df_task1_output = pd.read_csv(main_dir / 'task_1' / 'outputs_task1' / 'csv_outputs' / 'output_one_per_iteration_tol_0.01_seed_123.csv', header=None)

    # extract data needed
    tems = inp_dfe.loc[:, 'tavg__ref'].values
    ppts = inp_dfe.loc[:, 'pptn__ref'].values
    change_values = df_change_value_inputs.loc[:, 'change_value_inputs'].values
    perturbed_ofv = 1 - df_change_value_inputs.loc[:, 'new_ofv_old_params'].values
    recalibrated_ofv = 1 - df_change_value_inputs.loc[:, 'new_opt_ofv'].values
    prm_optimized = df_task1_output[1].iloc[-1]
    prm_names = df_cumulated_output['changed_parameter']

    # plots. the next 3 lines of code work only for the small data set because read error of change values
    # TODO
    plot_perturbed_time_series()
    plot_ofv_vs_change(perturbed_ofv, opt=True)
    plot_ofv_vs_change(recalibrated_ofv, opt=False)

    # box plots
    new_opt_prms = df_change_value_inputs['new_opt_params'].to_numpy()
    new_opt_prms = np.array([np.fromstring(row.strip('[]'), sep=' ') for row in new_opt_prms]).T.tolist()

    box_plot_all(new_opt_prms, prm_names)

    for i in np.arange(len(prm_names)):
        box_plot_prm(prm_names[i], new_opt_prms[i], prm_optimized[i])

    # violin plots
    violin_plot_all(new_opt_prms, prm_names)

    for i in np.arange(len(prm_names)):
        violin_plot_prm(prm_names[i], new_opt_prms[i], prm_optimized[i])

    # create box and violin plots without lrr_tdh
    prm_names = np.delete(prm_names, 20)
    new_opt_prms_pop = new_opt_prms.pop(20)

    box_plot_all(new_opt_prms, prm_names, True)
    violin_plot_all(new_opt_prms, prm_names, True)

    print("script finished without errors")
