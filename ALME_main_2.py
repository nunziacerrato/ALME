import os
import time
import numpy as np
import pandas as pd
import qutip
import tenpy
from lindblad import *
from ALME import *
import matplotlib.pylab as plt

t0, Dt = 0, 0.001
iterations = 2000

Kraus = True
Standard = True
save_time_N = True
alpha, gamma = 1, 1

N_list = [2,3,4,5,6,7,8]

dict_time_N = {'N' : [], 'elaps_approx_Kraus' : [], 'elaps_stand' : []}

for N in N_list:
    dict_time_N['N'].append(N)
    dict_data = {'t_approx_Kraus' : [], 't_standard' : []}

    elapsed_time_approx_Kraus = 0
    elapsed_time_standard = 0

    for j in range(iterations):
        
        RM_D = np.array(qutip.rand_dm_ginibre(N**2 -1, rank=None))
        L, L_dagg = create_L_L_dagg(N,RM_D)
        
        L_new = np.zeros((N**2-1, N**2, N**2), dtype=complex)
        L_new_dagg = np.zeros((N**2-1, N**2, N**2), dtype=complex)
        for i in range(N**2-1):
            L_new[i] = np.kron(L[i], np.eye(N))
            L_new_dagg[i] = np.kron((np.conjugate(L[i])).T, np.eye(N))
        
        ham = tenpy.linalg.random_matrix.GUE((N,N))
        ham_kron = np.kron(ham, np.eye(N))

        if Kraus == True:

            state = max_ent(N)
            neg = neg_ent(state,N)
            t = t0

            t_in_approx_Kraus = time.time()
            while(neg >= 1e-13):

                state = MP_IIord(ham_kron,L_new,Dt,state)
                t = t + Dt
                neg = neg_ent(state,N)
                if np.abs(np.real(neg)) < 1e-13:
                    dict_data['t_approx_Kraus'].append(t)
                    print(f'iter = {j}, t_ent_approx_Kraus = {t}')
            t_fin_approx_Kraus = time.time()
            elapsed_time_approx_Kraus = elapsed_time_approx_Kraus + \
                                        (t_fin_approx_Kraus - t_in_approx_Kraus)    

        if Standard == True:  
            state = max_ent(N)
            neg = neg_ent(state,N)
            t = t0
            
            Lind_matr = Lindbladian_matrix(N,RM_D,ham,alpha,gamma)

            t_in_standard = time.time()
            while(neg >= 1e-13):

                neg = negat_ent(N,Lind_matr,t)
                if np.abs(np.real(neg)) < 1e-13:
                    dict_data['t_standard'].append(t)
                    print(f'iter = {j}, t_ent_standard = {t}')
                t = t + Dt
            t_fin_standard = time.time()
            elapsed_time_standard = elapsed_time_standard + (t_fin_standard - t_in_standard)

    # Save the t_ent data in an Excel file.
    df_data = pd.concat([pd.Series(dict_data[key], name = f'{key}')\
                        for key in dict_data.keys()], axis = 1)
    dir = os.getcwd()
    writer = pd.ExcelWriter(f'{dir}\\ALME_data\\2_t_ent_N_{N}_{iterations}_iterations_Dt={Dt}.xlsx')
    df_data.to_excel(writer, 'DataFrame')
    writer.close()
    print(f'DATA FILE CORRECTLY SAVED FOR {iterations} ITERATIONS - N = {N}')

    # Save the elapsed times for each dimension N at the corresponding dictionary keys.
    if save_time_N == True:
        if Kraus == True:
            dict_time_N['elaps_approx_Kraus'].append(elapsed_time_approx_Kraus)
        if Standard == True:
            dict_time_N['elaps_stand'].append(elapsed_time_standard)

# Save the elapsed times for each dimension N in an Excel file
if save_time_N == True:
    df_time_N = pd.concat([pd.Series(dict_time_N[key], name = f'{key}')\
                           for key in dict_time_N.keys()], axis = 1)
    dir = os.getcwd()
    writer_times_N = \
            pd.ExcelWriter(f'{dir}\\ALME_data\\22_times_for_N_t_ent_iters={iterations}_Dt={Dt}.xlsx')
    df_time_N.to_excel(writer_times_N, 'DataFrame')
    writer_times_N.close()
    print(f'TIME N FILE CORRECTLY SAVED')



















# plt.hist(dict_data['t_approx_Kraus'], bins = 'auto', density = 'True', histtype='step', fill = False, label = 'Kraus')
# plt.hist(dict_data['t_standard'], bins = 'auto', density = 'True', histtype='step', fill = False, label = 'Standard')
# plt.legend()
# plt.show()






