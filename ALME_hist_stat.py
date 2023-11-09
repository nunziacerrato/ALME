import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import scipy.stats
import scipy.special
import scipy.spatial.distance

Dt = 0.01
N_list = [2,3]
iterations_list = [5000,3000]
N_iter_zip = list(zip(N_list,iterations_list))

path = os.getcwd()
dir = os.path.abspath(os.path.join(path, os.pardir))

iterations = 5000
# print(N_iter_zip)

for elem in N_iter_zip:
    N = elem[0]
    iterations = elem[1]
    df_fromexcel = pd.read_excel(f'{dir}\\ALME\\ALME_data\\' \
                                f't_ent_N_{N}_{iterations}_iterations_Dt={Dt}_II.xlsx')

    # Access to the column 't_ent' and drop NaN values if present
    t_ent_standard = df_fromexcel['t_standard'].dropna().values
    t_ent_Kraus = df_fromexcel['t_approx_Kraus'].dropna().values

    st_val, st_bins, st_patches = plt.hist(t_ent_standard, bins = 'auto', density = 'True',
                                        histtype='step', fill = False, label = 'Standard')
    Kr_val, Kr_bins, Kr_patches = plt.hist(t_ent_Kraus, bins = st_bins, density = 'True',
                                        histtype='step', fill = False, label = 'Kraus')
    
    # print('------- Stand -------')
    # print(st_val)
    # print('------- Kraus -------')
    # print(Kr_val)
    # print('------- Diff  -------')
    # diff = st_val - Kr_val
    # print(diff)
    prec = 6
    KL_div = np.round((scipy.stats.entropy(st_val, Kr_val, base=None)), prec)
    KL_div_special = np.round((scipy.special.kl_div(st_val, Kr_val, out=None)), prec)
    Jen_Shann = np.round(((scipy.spatial.distance.jensenshannon(st_val, Kr_val))), prec)
    kolmogorov_dist = (0.5)*sum(abs(st_val - Kr_val))
    print(f'N = {N}, dt = {Dt}')
    print(f'KL_div = {KL_div}')
    # print(f'KL_div_special = {KL_div_special}')
    print(f'JS_dist = {Jen_Shann}')
    print(f'Kolmogorov_dist = {kolmogorov_dist}')
