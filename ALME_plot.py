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
path = os.getcwd()
dir = os.path.abspath(os.path.join(path, os.pardir))

# Plot of the histograms obtained with the two approximated schemes and the standard method
if False:
    df_fromexcel_Kraus = pd.read_excel(f'{dir}\\UMEL\\UMEL_data\\' \
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
    # mean_t_ent_approx_Umel = np.round(np.mean(t_ent_approx_Umel),prec_t)
    # std_t_ent_approx_Umel = np.round(np.std(t_ent_approx_Umel),prec_t)
    # median_t_ent_approx_Umel = np.round(np.median(t_ent_approx_Umel),prec_t)
    # min_t_ent_approx_Umel = np.round(min(t_ent_approx_Umel),prec_t)


    # Plot the histogram of results
    fig_hist_Kraus, ax_hist_Kraus = plt.subplots(figsize=(15,10))
    
    # ax_hist_Kraus.hist(t_ent_standard, bins = 'auto', histtype='step', fill = False, density = True,\
    #             label = fr'Exact - Standard' + '\n' + fr'$\overline{{ \tau}}_{{ent}}^{{st}}$ = '\
    #             fr'{mean_t_ent_standard}, $\sigma(\tau_{{ent}}^{{st}})$'\
    #             fr'= {std_t_ent_standard}, Me($\tau_{{ent}}^{{st}}$) = {median_t_ent_standard}, '\
    #             fr'min($\tau_{{ent}}^{{st}}$) = {min_t_ent_standard}')
    # ax_hist_Kraus.hist(t_ent_approx_Kraus, bins = 'auto', histtype='step', fill = False,\
    #             density = True, label = fr'Approx - Kraus scheme' + '\n' + \
    #             fr'$\overline{{ \tau}}_{{ent}}^{{Kraus}}$ = {mean_t_ent_approx_Kraus}, '\
    #             fr'$\sigma(\tau_{{ent}}^{{Kraus}})$ = {std_t_ent_approx_Kraus}, '\
    #             fr'Me($\tau_{{ent}}^{{Kraus}}$) = {median_t_ent_approx_Kraus}, '\
    #             fr'min($\tau_{{ent}}^{{Kraus}}$) = {min_t_ent_approx_Kraus}')
    if N == 2:
        ax_hist_Kraus.set_title(r'$\bf{P}$$_{est}(x)$, 'fr'N = {N}, {iterations} iterations, dt = {Dt}',\
                            fontsize=35)
        ax_hist_Kraus.hist(t_ent_standard, bins = 'auto', histtype='step', fill = False, density = True,\
                    label = fr'Exact - Standard' + '\n' + \
                    fr'$\overline{{ x}}_{{ent}}^{{st}}$ = {mean_t_ent_standard}, '\
                    fr'$\sigma(x_{{ent}}^{{st}})$ = {std_t_ent_standard}, '\
                    fr'Me($x_{{ent}}^{{st}}$) = {median_t_ent_standard}, '\
                    fr'min($x_{{ent}}^{{st}}$) = {min_t_ent_standard}')
        ax_hist_Kraus.hist(t_ent_approx_Kraus, bins = 'auto', histtype='step', fill = False,\
                    density = True, label = fr'Approx - Kraus' + '\n' + \
                    fr'$\overline{{ x}}_{{ent}}^{{Kr}}$ = {mean_t_ent_approx_Kraus}, '\
                    fr'$\sigma(x_{{ent}}^{{Kr}})$ = {std_t_ent_approx_Kraus}, '
                    fr'Me($x_{{ent}}^{{Kr}}$) = {median_t_ent_approx_Kraus}, '\
                    fr'min($x_{{ent}}^{{Kr}}$) = {min_t_ent_approx_Kraus}')
    if N >= 3:
        ax_hist_Kraus.set_title(r'$\bf{P}$$_{ppt}(x)$, 'fr'N = {N}, {iterations} iterations, dt = {Dt}',\
                            fontsize=35)
        ax_hist_Kraus.hist(t_ent_standard, bins = 'auto', histtype='step', fill = False, density = True,\
                    label = fr'Exact - Standard' + '\n' + \
                    fr'$\overline{{ x}}_{{ppt}}^{{st}}$ = {mean_t_ent_standard}, '
                    fr'$\sigma(x_{{ppt}}^{{st}})$ = {std_t_ent_standard}, '\
                    fr'Me($x_{{ppt}}^{{st}}$) = {median_t_ent_standard}, '\
                    fr'min($x_{{ppt}}^{{st}}$) = {min_t_ent_standard}')
        ax_hist_Kraus.hist(t_ent_approx_Kraus, bins = 'auto', histtype='step', fill = False,\
                    density = True, label = fr'Approx - Kraus' + '\n' + \
                    fr'$\overline{{ x}}_{{ppt}}^{{Kr}}$ = {mean_t_ent_approx_Kraus}, '\
                    fr'$\sigma(x_{{ppt}}^{{Kr}})$ = {std_t_ent_approx_Kraus}, '\
                    fr'Me($x_{{ppt}}^{{Kr}}$) = {median_t_ent_approx_Kraus}, '\
                    fr'min($x_{{ppt}}^{{Kr}}$) = {min_t_ent_approx_Kraus}')
    
    # ax_hist_Kraus.hist(t_ent_approx_Umel, bins = 'auto', histtype='step', fill = False, \
    #             density = True, label = fr'Approx - Umel scheme' + '\n' + \
    #             fr'$\overline{{ \tau}}_{{ent}}^{{Umel}}$ = {mean_t_ent_approx_Umel}, '\
    #             fr' $\sigma(\tau_{{ent}}^{{Umel}})$ = {std_t_ent_approx_Umel},  '\
    #             fr'Me($\tau_{{ent}}^{{Umel}}$) = {median_t_ent_approx_Umel}, '\
    #             fr'min($\tau_{{ent}}^{{Umel}}$) = {min_t_ent_approx_Umel}')
    ax_hist_Kraus.set_xlabel(fr'$x$', fontsize = 30)
    ax_hist_Kraus.tick_params(labelsize=20)
    ax_hist_Kraus.legend(fontsize = 18)
    fig_hist_Kraus.savefig(f'{dir}\\ALME\\ALME_plot\\'
                f'Histograms_Kraus_stand_N={N}_{iterations}_iterations_Dt={Dt}.png', \
                bbox_inches='tight')
    plt.show()

# Plot of computational times with respect to the dimension N
if False:
    iterations = 5
    Dt = 0.01
    df_fromexcel_Kraus_time = pd.read_excel(f'{dir}\\ALME\\ALME_data\\'
                                            f'times_for_N_t_ent_iters={iterations}_Dt={Dt}_II_soloN=10.xlsx')
    
    # iters_N = df_fromexcel_Kraus_time['N']
    # t_stand = df_fromexcel_Kraus_time['elaps_stand']
    # t_approx_Kraus = df_fromexcel_Kraus_time['elaps_approx_Kraus']
    # t_approx_Umel = df_fromexcel_Kraus_time['elaps_approx_Umel']

    iters_N = df_fromexcel_Kraus_time['N'][0:5]
    t_stand = df_fromexcel_Kraus_time['elaps_stand'][0:5]
    t_approx_Kraus = df_fromexcel_Kraus_time['elaps_approx_Kraus'][0:5]

    fig_time, ax_time = plt.subplots(figsize=(15,10))
    ax_time.set_title(f'Elapsed Computational Time (s) vs Dimension', fontsize=25)
    ax_time.plot(iters_N, t_stand, label = 'Standard method')
    ax_time.plot(iters_N, t_approx_Kraus, label = 'Kraus method')
    # ax_time.plot(iters_N, t_approx_Umel, label = 'Approx Umel')
    ax_time.set_xlabel('N', fontsize=25)
    ax_time.legend(fontsize = 20)
    ax_time.tick_params(labelsize=20)
    # fig_time.savefig(f'{dir}\\ALME\\ALME_plot\\Kraus_elapsed_time_{iterations}_iterations_Dt={Dt}.png', \
    #                 bbox_inches='tight')
    plt.show()
    

# Plot of the histograms as N varies
if True:

    # Plot the histogram of results
    fig_hist_Kraus, ax_hist_Kraus = plt.subplots(figsize=(15,10))

    N_list = [2,3,4,5,6]
    iterations_list = [5000,3000,3000,3000,1500]
    N_iter_zip = list(zip(N_list,iterations_list))
    Dt = 0.001

    for elem in N_iter_zip:
        N = elem[0]
        iterations = elem[1]
        df_fromexcel_Kraus = pd.read_excel(f'{dir}\\ALME\\ALME_data\\' \
                                            f't_ent_N_{N}_{iterations}_iterations_Dt={Dt}_II.xlsx')

        # Access to the column 't_ent' and drop NaN values if present
        t_ent_standard = df_fromexcel_Kraus['t_standard'].dropna().values
        t_ent_approx_Kraus = df_fromexcel_Kraus['t_approx_Kraus'].dropna().values

        # Compute mean, standard deviation and median of standard t_ent values
        # mean_t_ent_standard = np.round(np.mean(t_ent_standard),prec_t)
        # std_t_ent_standard = np.round(np.std(t_ent_standard),prec_t)
        # median_t_ent_standard = np.round(np.median(t_ent_standard),prec_t)
        # min_t_ent_standard = np.round(min(t_ent_standard),prec_t)

        # Compute mean, standard deviation and median of approximate (Kraus) t_ent values
        mean_t_ent_approx_Kraus = np.round(np.mean(t_ent_approx_Kraus),prec_t)
        std_t_ent_approx_Kraus = np.round(np.std(t_ent_approx_Kraus),prec_t)
        median_t_ent_approx_Kraus = np.round(np.median(t_ent_approx_Kraus),prec_t)
        min_t_ent_approx_Kraus = np.round(min(t_ent_approx_Kraus),prec_t)

        ax_hist_Kraus.set_title(r'$\bf{P}$$_{ppt}(x)$, 'fr'histograms as N varies, {iterations} iterations, dt = {Dt}',\
                            fontsize=35)
        # ax_hist_Kraus.hist(t_ent_standard, bins = 'auto', histtype='step', fill = False, density = True,\
        #             label = fr'Standard, N = {N}, ' + \
        #             fr'$\overline{{x}}_{{ppt}}^{{st}}$ = {mean_t_ent_standard}')
                    # fr'$\sigma(x_{{ppt}}^{{st}})$ = {std_t_ent_standard}, '\
                    # fr'Me($x_{{ppt}}^{{st}}$) = {median_t_ent_standard}, '\
                    # fr'min($x_{{ppt}}^{{st}}$) = {min_t_ent_standard}')
        ax_hist_Kraus.hist(t_ent_approx_Kraus, bins = 'auto', histtype='step', fill = False,\
                    density = True, label = fr'Kraus, N = {N}, ' + \
                    fr'$\overline{{ x}}_{{ppt}}^{{Kr}}$ = {mean_t_ent_approx_Kraus}')
                    # fr'$\sigma(x_{{ppt}}^{{Kr}})$ = {std_t_ent_approx_Kraus}, '\
                    # fr'Me($x_{{ppt}}^{{Kr}}$) = {median_t_ent_approx_Kraus}, '\
                    # fr'min($x_{{ppt}}^{{Kr}}$) = {min_t_ent_approx_Kraus}')

    ax_hist_Kraus.set_xlabel(fr'$x$', fontsize = 30)
    ax_hist_Kraus.tick_params(labelsize=20)
    ax_hist_Kraus.legend(fontsize = 18)
    # fig_hist_Kraus.savefig(f'{dir}\\ALME\\ALME_plot\\'
    #             f'Histograms_as_N_varies_Dt={Dt}.png', \
    #             bbox_inches='tight')
    plt.show()

