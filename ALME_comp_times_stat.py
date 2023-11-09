import os
import time
import numpy as np
import pandas as pd
import qutip
import tenpy
from lindblad import *
from ALME import *
import matplotlib.pyplot as plt
import scipy.stats

t, Dt = 0, 0.001
N_list = [2,3,4,5,6,7,8,9,10] 
iterations = 10
save_time_N = False
Standard = True

path = os.getcwd()
dir = os.path.abspath(os.path.join(path, os.pardir))

dict_time_N = {'N' : [], 'elaps_approx_Kraus' : [], 'elaps_approx_Umel' : [], 'elaps_stand' : []}

# Cycle on the dimensions and the iteration number
for N in N_list:
    print(N)    
    dict_time_N['N'].append(N)
        
    elapsed_time_approx_Kraus = 0
    elapsed_time_standard = 0
    
    # Cycle on the iterations
    for j in range(iterations):

        # Sample random matrices: Lindblad operator and Hamiltonian
        RM_D = np.array(qutip.rand_dm_ginibre(N**2 -1, rank=None))
        L, L_dagg = create_L_L_dagg(N,RM_D)
        L_new = np.zeros((N**2-1, N**2, N**2), dtype=complex)
        L_new_dagg = np.zeros((N**2-1, N**2, N**2), dtype=complex)
        for i in range(N**2-1):
            L_new[i] = np.kron(L[i], np.eye(N))
            L_new_dagg[i] = np.kron((np.conjugate(L[i])).T, np.eye(N))
        ham = tenpy.linalg.random_matrix.GUE((N,N))
        ham_kron = np.kron(ham, np.eye(N))

        # Initialize the state at each iteration
        state = max_ent(N)
        neg = neg_ent(state,N)
        t, Dt = 0, 0.001

        ############################# KRAUS APPROXIMATION SCHEME #############################

        # Cycle until the condition on the negativity is satisfied
        t_in_approx_Kraus = time.time()
        while (neg >= 1e-13):

            # Compute the state after the interaction with the environment and its negativity
            state = MP_IIord(ham_kron,L_new,Dt,state)
            neg = neg_ent(state,N)

            # If the negativity is equal to zero, within the chosen tolerance, save the
            # corresponding t value as t_ent time (entanglement surivival time)
            if np.abs(np.real(neg)) < 1e-13:
                # print(f'Kraus neg = {neg}')
                print(f'iter = {j}, t_ent_approx_Kraus = {t}')
            # else:
            #     neg_old = neg
            t = t + Dt
        # print(f'Kraus neg old = {neg_old}')   
        t_fin_approx_Kraus = time.time()
        elapsed_time_approx_Kraus = elapsed_time_approx_Kraus + \
                                    (t_fin_approx_Kraus - t_in_approx_Kraus)

        ########################### STANDARD SCHEME - LIOUVILLE REP ###########################
        if Standard == True:
            # Define parameters for the standard case
            alpha, gamma = 1, 1
            t, t_max, Dt = 0, 10, 0.001

            # Compute the matrix associated to the Lindblad superoperator
            Lind_matr = Lindbladian_matrix(N,RM_D,ham,alpha,gamma)
            
            # Cycle until the condition is satisfied and save the t_ent time
            t_in_standard = time.time()
            while (t < t_max):
                n_ent = negat_ent(N,Lind_matr,t)
                if np.abs(np.real(n_ent)) < 1e-13:
                    # print(f'Stand neg = {n_ent}')
                    print(f'Stand - iter = {j}, t_ent_standard = {t}')
                    break
                # else:
                #     neg_old = n_ent
                t = t + Dt
            # print(f'Standard neg old = {neg_old}')   
            t_fin_standard = time.time()
            elapsed_time_standard = elapsed_time_standard + (t_fin_standard - t_in_standard)

        ######################### ADAPTIVE STANDARD SCHEME - LIOUVILLE REP #########################
        if False:
            # Define parameters for the standard case
            alpha, gamma = 1, 1
            t, t_max, Dt = 0, 10, 0.5

            # Compute the matrix associated to the Lindblad superoperator
            Lind_matr = Lindbladian_matrix(N,RM_D,ham,alpha,gamma)

            # n_ent = negat_ent(N,Lind_matr,t)
            t_prec = t
            t_new = t + Dt
            t_in_standard = time.time()
            while (t_new < t_max):

                n_ent = negat_ent(N,Lind_matr,t_new)
                
                # print(f't_new = {t_new}, t_prec = {t_prec}')
                # print(f'n_ent = {n_ent}')

                if np.abs(np.real(n_ent)) <= 1e-13 and (t_new-t_prec) < 1e-4:
                    print(f'Adapt - iter = {j}, t_ent_standard = {t_new}')
                    t_new = np.round(t_new, 3)
                    print(f'Adapt - iter = {j}, t_ent_standard = {t_new}')
                    break
                if np.abs(np.real(n_ent)) <= 1e-13 and (t_new-t_prec) >= 1e-4:
                    # t_prec = t_new - Dt
                    t_new = t_prec
                    # print(f't_prec = {t_prec}, t_new = {t_new}')
                    Dt = Dt/2
                    # print(f'Dt = {Dt}')
                    # if Dt < 0.01:
                    #     Dt = 0.001
                    t_new = t_new + Dt
                    # print(f't_new = {t_new}')
                if np.abs(np.real(n_ent)) > 1e-13:
                    t_prec = t_new
                    t_new = t_prec + Dt  
            t_fin_standard = time.time()
            elapsed_time_standard = elapsed_time_standard + (t_fin_standard - t_in_standard)

        ########################################################################################
    
    # Save the elapsed times for each dimension N at the corresponding dictionary keys.
    # N.B. This part is here because we choose a fixed (small) iteration number when we are 
    # interested in saving the elapsed time for each dimension. This because this task requires
    # a huge computational effort.
    dict_time_N['elaps_approx_Kraus'].append(elapsed_time_approx_Kraus)
    if Standard == True:
        dict_time_N['elaps_stand'].append(elapsed_time_standard)

# Save the elapsed times for each dimension N in an Excel file
df_time_N = pd.concat([pd.Series(dict_time_N[key], name = f'{key}')\
                        for key in dict_time_N.keys()], axis = 1)
dir = os.getcwd()
writer_times_N = \
        pd.ExcelWriter(f'{dir}\\ALME_data\\times_for_N_t_ent_iters={iterations}_Dt={Dt}_II.xlsx')
                        # mode = 'a', if_sheet_exists = 'new')
df_time_N.to_excel(writer_times_N, 'DataFrame')
writer_times_N.close()
print(f'TIME N FILE CORRECTLY SAVED')




# df_fromexcel = pd.read_excel(f'{dir}\\ALME\\ALME_data\\'
#                              f'times_for_N_t_ent_iters={iterations}_Dt={Dt}_III.xlsx')

# comp_times_standard = df_fromexcel['elaps_stand'].dropna().values
# comp_times_Kraus = df_fromexcel['elaps_approx_Kraus'].dropna().values



