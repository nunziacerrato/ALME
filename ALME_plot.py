''' In this files we create the plots of the histograms of EST and PPT distributions and the plot
    of computational times with respect to the dimension (N) of one of the two subsystems of the
    system of interest.
'''
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set parameters for the plots
N = 2
Dt = 0.01
iterations = 5000
prec_t = 3
dir = os.getcwd()

# Plot of the histograms obtained with the two approximated schemes and the standard method
if True:
    df_fromexcel_Kraus = pd.read_excel(f'{dir}\\UMEL_data\\' \
                                       f't_ent_N_{N}_{iterations}_iterations_Dt={Dt}_II.xlsx')

    # Access to the column 't_ent' and drop NaN values if present
    t_ent_standard = df_fromexcel_Kraus['t_standard'].dropna().values
    t_ent_approx_Kraus = df_fromexcel_Kraus['t_approx_Kraus'].dropna().values
    t_ent_approx_Umel = df_fromexcel_Kraus['t_approx_Umel'].dropna().values

    # Compute mean, standard deviation and median of standard t_ent values
    mean_t_ent_standard = np.round(np.mean(t_ent_standard),prec_t)
    std_t_ent_standard = np.round(np.std(t_ent_standard),prec_t)
    median_t_ent_standard = np.round(np.median(t_ent_standard),prec_t)
    min_t_ent_standard = np.round(min(t_ent_standard),prec_t)

    # Compute mean, standard deviation and median of approximate (Kraus) t_ent values
    mean_t_ent_approx_Kraus = np.round(np.mean(t_ent_approx_Kraus),prec_t)
    std_t_ent_approx_Kraus = np.round(np.std(t_ent_approx_Kraus),prec_t)
    median_t_ent_approx_Kraus = np.round(np.median(t_ent_approx_Kraus),prec_t)
    min_t_ent_approx_Kraus = np.round(min(t_ent_approx_Kraus),prec_t)

    # Compute mean, standard deviation and median of approximate (Umel) t_ent values
    mean_t_ent_approx_Umel = np.round(np.mean(t_ent_approx_Umel),prec_t)
    std_t_ent_approx_Umel = np.round(np.std(t_ent_approx_Umel),prec_t)
    median_t_ent_approx_Umel = np.round(np.median(t_ent_approx_Umel),prec_t)
    min_t_ent_approx_Umel = np.round(min(t_ent_approx_Umel),prec_t)


    # Plot the histogram of results
    fig_hist_Kraus, ax_hist_Kraus = plt.subplots(figsize=(15,10))
    ax_hist_Kraus.set_title(r'$\bf{P}$$_{ent}(x)$, 'fr'N = {N}, {iterations} iterations',\
                            fontsize=35)
    ax_hist_Kraus.hist(t_ent_standard, bins = 'auto', histtype='step', fill = False, density = True,\
                label = fr'Standard' + '\n' + fr'$\overline{{ \tau}}_{{ent}}^{{st}}$ = '\
                fr'{mean_t_ent_standard}, $\sigma(\tau_{{ent}}^{{st}})$'\
                fr'= {std_t_ent_standard}, Me($\tau_{{ent}}^{{st}}$) = {median_t_ent_standard}, '\
                fr'min($\tau_{{ent}}^{{st}}$) = {min_t_ent_standard}')
    ax_hist_Kraus.hist(t_ent_approx_Kraus, bins = 'auto', histtype='step', fill = False,\
                density = True, label = fr'Approx - Kraus scheme' + '\n' + \
                fr'$\overline{{ \tau}}_{{ent}}^{{Kraus}}$ = {mean_t_ent_approx_Kraus}, '\
                fr'$\sigma(\tau_{{ent}}^{{Kraus}})$ = {std_t_ent_approx_Kraus}, '\
                fr'Me($\tau_{{ent}}^{{Kraus}}$) = {median_t_ent_approx_Kraus}, '\
                fr'min($\tau_{{ent}}^{{Kraus}}$) = {min_t_ent_approx_Kraus}')
    ax_hist_Kraus.hist(t_ent_approx_Umel, bins = 'auto', histtype='step', fill = False, \
                density = True, label = fr'Approx - Umel scheme' + '\n' + \
                fr'$\overline{{ \tau}}_{{ent}}^{{Umel}}$ = {mean_t_ent_approx_Umel}, '\
                fr' $\sigma(\tau_{{ent}}^{{Umel}})$ = {std_t_ent_approx_Umel},  '\
                fr'Me($\tau_{{ent}}^{{Umel}}$) = {median_t_ent_approx_Umel}, '\
                fr'min($\tau_{{ent}}^{{Umel}}$) = {min_t_ent_approx_Umel}')
    ax_hist_Kraus.set_xlabel(fr'$x$', fontsize = 30)
    ax_hist_Kraus.tick_params(labelsize=20)
    ax_hist_Kraus.legend(fontsize = 15)
    fig_hist_Kraus.savefig(f'{dir}\\UMEL_plot\\'
                f'NEW_Histograms_all_approx_stand_N={N}_{iterations}_iterations_Dt={Dt}.png', \
                bbox_inches='tight')
    plt.show()

# Plot of computational times with respect to the dimension N
if True:
    iterations = 5
    Dt = 0.01
    df_fromexcel_Kraus_time = pd.read_excel(f'{dir}\\UMEL_data\\'
                                            f'times_for_N_t_ent_iters={iterations}_Dt={Dt}_II.xlsx')
    
    iters_N = df_fromexcel_Kraus_time['N']
    t_stand = df_fromexcel_Kraus_time['elaps_stand']
    t_approx_Kraus = df_fromexcel_Kraus_time['elaps_approx_Kraus']
    t_approx_Umel = df_fromexcel_Kraus_time['elaps_approx_Umel']

    fig_time, ax_time = plt.subplots(figsize=(15,10))
    ax_time.set_title(f'Elapsed time vs dimension', fontsize=25)
    ax_time.plot(iters_N, t_stand, label = 'Standard')
    ax_time.plot(iters_N, t_approx_Kraus, label = 'Approx Kraus')
    ax_time.plot(iters_N, t_approx_Umel, label = 'Approx Umel')
    ax_time.set_xlabel('N', fontsize=20)
    ax_time.legend(fontsize = 18)
    ax_time.tick_params(labelsize=20)
    fig_time.savefig(f'{dir}\\UMEL_plot\\NEW_elapsed_time_{iterations}_iterations_Dt={Dt}.png', \
                    bbox_inches='tight')
    plt.show()
    

