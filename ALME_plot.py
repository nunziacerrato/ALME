''' In this files we create the plots of the histograms of EST and PPT distributions and the plot
    of computational times with respect to the dimension (N) of one of the two subsystems of the
    system of interest.
'''
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set parameters for the plots
N = 5
Dt = 0.001
iterations = 2000
prec_t = 3
path = os.getcwd()
dir = os.path.abspath(os.path.join(path, os.pardir))

# Plot of the histograms obtained with the two approximated schemes and the standard method
if False:
    df_fromexcel_Kraus = pd.read_excel(f'{dir}\\ALME\\ALME_data\\' \
                                       f'2_t_ent_N_{N}_{iterations}_iterations_Dt={Dt}.xlsx')

    # Access to the column 't_ent' and drop NaN values if present
    t_ent_standard = df_fromexcel_Kraus['t_standard'].dropna().values
    t_ent_approx_Kraus = df_fromexcel_Kraus['t_approx_Kraus'].dropna().values

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


    # Plot the histogram of results
    fig_hist_Kraus, ax_hist_Kraus = plt.subplots(figsize=(15,10))
   
    if N == 2:
        ax_hist_Kraus.set_title(r'$\bf{P}$$_{est}(x)$, 'fr'N = {N}, {iterations} iterations, dt = {Dt}',\
                            fontsize=35)
        ax_hist_Kraus.hist(t_ent_standard, bins = 'auto', histtype='step', fill = False, density = True,\
                    label = fr'Exact - Standard' + '\n' + \
                    fr'$\overline{{ x}}_{{ent}}^{{st}}$ = {mean_t_ent_standard}, '\
                    fr'$\sigma(x_{{ent}}^{{st}})$ = {std_t_ent_standard}, ' + '\n' + \
                    fr'Me($x_{{ent}}^{{st}}$) = {median_t_ent_standard}, '\
                    fr'min($x_{{ent}}^{{st}}$) = {min_t_ent_standard}')
        ax_hist_Kraus.hist(t_ent_approx_Kraus, bins = 'auto', histtype='step', fill = False,\
                    density = True, label = fr'Approx - Kraus' + '\n' + \
                    fr'$\overline{{ x}}_{{ent}}^{{Kr}}$ = {mean_t_ent_approx_Kraus}, '\
                    fr'$\sigma(x_{{ent}}^{{Kr}})$ = {std_t_ent_approx_Kraus}, ' + '\n' + \
                    fr'Me($x_{{ent}}^{{Kr}}$) = {median_t_ent_approx_Kraus}, '\
                    fr'min($x_{{ent}}^{{Kr}}$) = {min_t_ent_approx_Kraus}')
    if N >= 3:
        ax_hist_Kraus.set_title(r'$\bf{P}$$_{ppt}(x)$, 'fr'N = {N}, {iterations} iterations, dt = {Dt}',\
                            fontsize=35)
        ax_hist_Kraus.hist(t_ent_standard, bins = 'auto', histtype='step', fill = False, density = True,\
                    label = fr'Exact - Standard' + '\n' + \
                    fr'$\overline{{x}}_{{ppt}}^{{st}}$ = {mean_t_ent_standard}, '
                    fr'$\sigma(x_{{ppt}}^{{st}})$ = {std_t_ent_standard}, ' + '\n' + \
                    fr'Me($x_{{ppt}}^{{st}}$) = {median_t_ent_standard}, '\
                    fr'min($x_{{ppt}}^{{st}}$) = {min_t_ent_standard}')
        ax_hist_Kraus.hist(t_ent_approx_Kraus, bins = 'auto', histtype='step', fill = False,\
                    density = True, label = fr'Approx - Kraus' + '\n' + \
                    fr'$\overline{{x}}_{{ppt}}^{{Kr}}$ = {mean_t_ent_approx_Kraus}, '\
                    fr'$\sigma(x_{{ppt}}^{{Kr}})$ = {std_t_ent_approx_Kraus}, ' + '\n' + \
                    fr'Me($x_{{ppt}}^{{Kr}}$) = {median_t_ent_approx_Kraus}, '\
                    fr'min($x_{{ppt}}^{{Kr}}$) = {min_t_ent_approx_Kraus}')
    
    
    ax_hist_Kraus.set_xlabel(fr'$x$', fontsize = 30)
    ax_hist_Kraus.tick_params(labelsize=20)
    ax_hist_Kraus.legend(fontsize = 20, loc='upper right')
    fig_hist_Kraus.savefig(f'{dir}\\ALME\\ALME_plot\\'
                f'2_Histograms_Kraus_stand_N={N}_{iterations}_iterations_Dt={Dt}.png', \
                bbox_inches='tight')
    # plt.show()

# Plot of computational times with respect to the dimension N
if False:
    iterations = 1
    Dt = 0.001
    df_fromexcel_Kraus_time = pd.read_excel(f'{dir}\\ALME\\ALME_data\\'
                                            f'elapsed_times_optimized_alg.xlsx')
                                            # f'3_times_for_N_t_ent_iters={iterations}_Dt={Dt}.xlsx')
    
    iters_N = df_fromexcel_Kraus_time['N'][0:4]
    t_stand = df_fromexcel_Kraus_time['elaps_stand'][0:4]/60
    t_approx_Kraus = df_fromexcel_Kraus_time['elaps_approx_Kraus'][0:4]/60

    fig_time, ax_time = plt.subplots(figsize=(15,10))
    ax_time.set_title(f'Elapsed computational times vs dimension', fontsize=25)
    ax_time.scatter(iters_N, t_stand, label = 'Standard method', s = 100)
    ax_time.scatter(iters_N, t_approx_Kraus, label = 'Kraus method', s = 100)
    # ax_time.set_yscale('log')
    ax_time.set_xlabel('N', fontsize=25)
    ax_time.set_ylabel('Elapsed time (min)', fontsize=25)
    ax_time.legend(fontsize = 22)
    ax_time.tick_params(labelsize=20)
    fig_time.savefig(f'{dir}\\ALME\\ALME_plot\\Kraus_elapsed_time_{iterations}_iterations_Dt={Dt}_minn.png', \
                    bbox_inches='tight')
    plt.show()
    

# Plot of the histograms as N varies
if False:

    # Plot the histogram of results
    fig_hist_Kraus, ax_hist_Kraus = plt.subplots(figsize=(15,10))

    # N_list = [2,3,4,5,6,7]
    # iterations_list = [5000,3000,3000,3000,1500,1500]

    N_list = [2,3,4,5]
    iterations_list = [2000,2000,2000,2000]
    N_iter_zip = list(zip(N_list,iterations_list))
    Dt = 0.001

    for elem in N_iter_zip:
        N = elem[0]
        iterations = elem[1]
        df_fromexcel_Kraus = pd.read_excel(f'{dir}\\ALME\\ALME_data\\' \
                                            f'2_t_ent_N_{N}_{iterations}_iterations_Dt={Dt}.xlsx')

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

        ax_hist_Kraus.set_title(r'$\bf{P}$$_{ppt}(x)$ as $N$ varies',\
                            fontsize=35)
        # ax_hist_Kraus.hist(t_ent_standard, bins = 'auto', histtype='step', fill = False, density = True,\
        #             label = fr'Standard, N = {N}, ' + \
        #             fr'$\overline{{x}}_{{ppt}}^{{st}}$ = {mean_t_ent_standard}')
                    # fr'$\sigma(x_{{ppt}}^{{st}})$ = {std_t_ent_standard}, '\
                    # fr'Me($x_{{ppt}}^{{st}}$) = {median_t_ent_standard}, '\
                    # fr'min($x_{{ppt}}^{{st}}$) = {min_t_ent_standard}')
        ax_hist_Kraus.hist(t_ent_approx_Kraus, bins = 'auto', histtype='step', fill = False,\
                    density = True, label = fr'N = {N}')
                    # , ' + \
                    # fr'$\overline{{x}}_{{ppt}}^{{Kr}}$ = {mean_t_ent_approx_Kraus}')
                    # fr'$\sigma(x_{{ppt}}^{{Kr}})$ = {std_t_ent_approx_Kraus}, '\
                    # fr'Me($x_{{ppt}}^{{Kr}}$) = {median_t_ent_approx_Kraus}, '\
                    # fr'min($x_{{ppt}}^{{Kr}}$) = {min_t_ent_approx_Kraus}')
    
    for N in [6,7]:
        iterations = 1500
        df_fromexcel_Kraus_II = pd.read_excel(f'{dir}\\ALME\\ALME_data\\' \
                                            f't_ent_N_{N}_{iterations}_iterations_Dt={Dt}_II.xlsx')
        t_ent_standard = df_fromexcel_Kraus_II['t_standard'].dropna().values
        t_ent_approx_Kraus = df_fromexcel_Kraus_II['t_approx_Kraus'].dropna().values + Dt
        ax_hist_Kraus.hist(t_ent_approx_Kraus, bins = 'auto', histtype='step', fill = False,\
                        density = True, label = fr'N = {N}')

    ax_hist_Kraus.set_xlabel(fr'$x$', fontsize = 30)
    ax_hist_Kraus.tick_params(labelsize=20)
    ax_hist_Kraus.legend(fontsize = 18)
    fig_hist_Kraus.savefig(f'{dir}\\ALME\\ALME_plot\\2_Histograms_as_N_varies_Dt={Dt}.png', \
                bbox_inches='tight')
    plt.show()



# Plot of the characteristic times of the histograms as N varies
if True:

    # Plot the histogram of results
    fig_Kraus, ax_Kraus = plt.subplots(figsize=(15,10))

    N_list = [2,3,4,5,6,7]
    iterations_list = [2000,2000,2000,2000,1500,1500]

    N_iter_zip = list(zip(N_list,iterations_list))
    Dt = 0.001
    mean_t = []
    median_t = []
    min_t = []
    std_t = []

    for elem in N_iter_zip:
        N = elem[0]
        iterations = elem[1]
        df_fromexcel_Kraus = pd.read_excel(f'{dir}\\ALME\\ALME_data\\' \
                                            f'2_t_ent_N_{N}_{iterations}_iterations_Dt={Dt}.xlsx')

        # Access to the column 't_ent' and drop NaN values if present
        t_ent_approx_Kraus = df_fromexcel_Kraus['t_approx_Kraus'].dropna().values
        if N == 6 or N == 7:
            t_ent_approx_Kraus = t_ent_approx_Kraus + Dt

        # Compute mean, standard deviation and median of approximate (Kraus) t_ent values
        mean_t.append(np.mean(t_ent_approx_Kraus))
        std_t.append(np.std(t_ent_approx_Kraus))
        median_t.append(np.median(t_ent_approx_Kraus))
        min_t.append(min(t_ent_approx_Kraus))

    
    ax_Kraus.scatter(N_list, mean_t, label = fr'mean')
    ax_Kraus.scatter(N_list, median_t, label = fr'median')
    # ax_Kraus.scatter(N_list, min_t, label = fr'min')
    # ax_Kraus.scatter(N_list, std_t, label = fr'std')

    ax_Kraus.set_title(fr'Characteristic (adim) times vs N - dt = {Dt}', fontsize=35)
    ax_Kraus.set_xlabel(fr'$N$', fontsize = 30)
    ax_Kraus.set_ylabel(fr'$x$', fontsize = 30)
    ax_Kraus.tick_params(labelsize=20)
    ax_Kraus.legend(fontsize = 18)
    # fig_Kraus.savefig(f'{dir}\\ALME\\ALME_plot\\Charact_times_N_varies_Dt={Dt}.png', \
    #             bbox_inches='tight')
    print(mean_t)
    print(median_t)
    # plt.show()

