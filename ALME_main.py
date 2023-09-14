import os
import time
import numpy as np
import pandas as pd
import qutip
import tenpy
import matplotlib.pyplot as plt
from lindblad import *
from ALME import *


# Define time parameters
t, Dt = 0, 0.001      # Starting time and time step

save_time_N = False
Umel = False
Standard = False

# Initialize lists and dictionaries
N_list = [6]#[2,3,4,5,6,7]
iterations_list = [3000]

t_ent_Kraus = []
t_ent_Umel = []
t_ent_standard = []

dict_data = {'t_approx_Kraus' : [], 't_approx_Umel' : [], 't_standard' : []}
dict_time = {'iters' : [], 'elaps_approx_Kraus' : [], 'elaps_approx_Umel' : [], 'elaps_stand' : []}
dict_time_N = {'N' : [], 'elaps_approx_Kraus' : [], 'elaps_approx_Umel' : [], 'elaps_stand' : []}

# Cycle on the dimensions and the iteration number
for N in N_list:
    print(N)
    dict_time_N['N'].append(N)

    for iterations in iterations_list:
        dict_time['iters'].append(iterations)
        
        elapsed_time_approx_Kraus = 0
        elapsed_time_approx_Umel = 0
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
            while (neg > 1e-12):

                # Compute the state after the interaction with the environment and its negativity
                state = MP_IIord(ham_kron,L_new,Dt,state)
                neg = neg_ent(state,N)

                # If the negativity is equal to zero, within the chosen tolerance, save the
                # corresponding t value as t_ent time (entanglement surivival time)
                if np.abs(np.real(neg)) < 1e-13:
                    t_ent_Kraus.append(t)
                    dict_data['t_approx_Kraus'].append(t)
                    print(f'iter = {j}, t_ent_approx_Kraus = {t}')
                t = t + Dt       
            t_fin_approx_Kraus = time.time()
            elapsed_time_approx_Kraus = elapsed_time_approx_Kraus + \
                                        (t_fin_approx_Kraus - t_in_approx_Kraus)

            ########################### STANDARD SCHEME - LIOUVILLE REP ###########################
            if Standard == True:
                # Define parameters for the standard case
                alpha, gamma = 1, 1
                t, t_max = 0, 10
        
                # Compute the matrix associated to the Lindblad superoperator
                Lind_matr = Lindbladian_matrix(N,RM_D,ham,alpha,gamma)
                
                # Cycle until the condition is satisfied and save the t_ent time
                t_in_standard = time.time()
                while (t < t_max):
                    n_ent = negat_ent(N,Lind_matr,t)
                    if np.abs(np.real(n_ent)) < 1e-13:
                        t_ent_standard.append(t)
                        dict_data['t_standard'].append(t)
                        print(f'iter = {j}, t_ent_standard = {t}')
                        break
                    t += Dt
                t_fin_standard = time.time()
                elapsed_time_standard = elapsed_time_standard + (t_fin_standard - t_in_standard)


            ############################# UMEL APPROXIMATION SCHEME #############################
            if Umel == True:
                # Initialize the state at each iteration
                state = max_ent(N)
                neg = neg_ent(state,N)
                t, Dt = 0, 0.001
                
                # Cycle until the condition on the negativity is satisfied
                t_in_approx_Umel = time.time()
                while (neg > 1e-12):

                    # Compute the state after the interaction with the environment and its negativity
                    state = unraveling_LME_II(state,ham_kron,L_new,L_new_dagg,Dt,N)
                    neg = neg_ent(state,N)

                    # If the negativity is equal to zero, within the chosen tolerance, save the
                    # corresponding t value as t_ent time (entanglement surivival time)
                    if np.abs(np.real(neg)) < 1e-13:
                        t_ent_Umel.append(t)
                        dict_data['t_approx_Umel'].append(t)
                        print(f'iter = {j}, t_ent_approx_Umel = {t}')
                    t = t + Dt       
                t_fin_approx_Umel = time.time()
                elapsed_time_approx_Umel = elapsed_time_approx_Umel + \
                                            (t_fin_approx_Umel - t_in_approx_Umel)

            ########################################################################################

        # Save the elapsed times for each iteration number at the corresponding dictionary keys
        dict_time['elaps_approx_Kraus'].append(elapsed_time_approx_Kraus)
        if Standard == True:
            dict_time['elaps_stand'].append(elapsed_time_standard)
        if Umel == True:
            dict_time['elaps_approx_Umel'].append(elapsed_time_approx_Umel)

        # Save the t_ent data in an Excel file.
        df_data = pd.concat([pd.Series(dict_data[key], name = f'{key}')\
                            for key in dict_data.keys()], axis = 1)
        dir = os.getcwd()
        writer = pd.ExcelWriter(f'{dir}\\UMEL_data\\t_ent_N_{N}_{iterations}_iterations_Dt={Dt}_II.xlsx')
        df_data.to_excel(writer, 'DataFrame')
        writer.close()
        print(f'DATA FILE CORRECTLY SAVED FOR {iterations} ITERATIONS - N = {N}')

        # Save the elapsed times for each iteration number at the corresponding dictionary keys.
        # This part is executed only when we are not interested in saving the elapsed time for each
        # dimension N. This because otherwise the file will be overwritten each time.
        if save_time_N == False:
            df_time_data = pd.concat([pd.Series(dict_time[key], name = f'{key}')\
                                    for key in dict_time.keys()], axis = 1)
            dir = os.getcwd()
            writer_times = \
                    pd.ExcelWriter(f'{dir}\\UMEL_data\\times_for_iters_t_ent_N_{N}_Dt={Dt}.xlsx')
            df_time_data.to_excel(writer_times, 'DataFrame')
            writer_times.close()
            print(f'TIME FILE CORRECTLY SAVED')
        
        # Save the elapsed times for each dimension N at the corresponding dictionary keys.
        # N.B. This part is here because we choose a fixed (small) iteration number when we are 
        # interested in saving the elapsed time for each dimension. This because this task requires
        # a huge computational effort.
        if save_time_N == True:
            dict_time_N['elaps_approx_Kraus'].append(elapsed_time_approx_Kraus)
            if Standard == True:
                dict_time_N['elaps_stand'].append(elapsed_time_standard)
            if Umel == True:
                dict_time_N['elaps_approx_Umel'].append(elapsed_time_approx_Umel)

# Save the elapsed times for each dimension N in an Excel file
if save_time_N == True:
    df_time_N = pd.concat([pd.Series(dict_time_N[key], name = f'{key}')\
                           for key in dict_time_N.keys()], axis = 1)
    dir = os.getcwd()
    writer_times_N = \
            pd.ExcelWriter(f'{dir}\\UMEL_data\\times_for_N_t_ent_iters={iterations}_Dt={Dt}_II.xlsx')
    df_time_N.to_excel(writer_times_N, 'DataFrame')
    writer_times_N.close()
    print(f'TIME N FILE CORRECTLY SAVED')
        

# t_stand = dict_time_N['elaps_stand']
# t_approx_Kraus = dict_time_N['elaps_approx_Kraus']
# t_approx_Umel = dict_time_N['elaps_approx_Umel']
# iters_N = dict_time_N['N']
# plt.plot(iters_N, t_stand, label = 'Standard')
# plt.plot(iters_N, t_approx_Kraus, label = 'Approx Kraus')
# plt.plot(iters_N, t_approx_Umel, label = 'Approx Umel')
# # plt.ylabel('t')
# plt.title('Elapsed time w.r.t the dimension of the system', fontsize=25)
# plt.xlabel('N', fontsize=18)
# plt.legend(fontsize = 18)
# plt.show()

# Plot the histogram of results
plt.title(r'$\bf{P}$$_{ent}(x)$, 'fr'N = {N}, {iterations} iterations', fontsize=35)
plt.hist(dict_data['t_approx_Kraus'], bins = 'auto', histtype='step', fill = False, density = True, label = 'Kraus')
if Standard == True:
    plt.hist(dict_data['t_standard'], bins = 'auto', histtype='step', fill = False, density = True, label = 'Standard')
if Umel == True:
    plt.hist(dict_data['t_approx_Umel'], bins = 'auto', histtype='step', fill = False, density = True, label = 'Umel')
plt.xlabel(fr'$x$', fontsize = 30)
plt.legend()
plt.show()


