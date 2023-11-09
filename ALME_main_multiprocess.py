import os
import time
import numpy as np
import pandas as pd
import qutip
import tenpy
import multiprocessing
from lindblad import *
from ALME import *


def ALME_savedata(N):

    # Define time parameters
    t, Dt = 0, 0.001      # Starting time and time step
    iterations = 1500

    Standard = False

    # Cycle on the dimensions and the iteration number
    dict_data = {'t_approx_Kraus' : [], 't_approx_Umel' : [], 't_standard' : []}
        
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
                dict_data['t_approx_Kraus'].append(t)
                print(f'N = {N}, iter = {j}, t_ent_approx_Kraus = {t}')
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
            t, t_max = 1.2, 10
    
            # Compute the matrix associated to the Lindblad superoperator
            Lind_matr = Lindbladian_matrix(N,RM_D,ham,alpha,gamma)
            
            # Cycle until the condition is satisfied and save the t_ent time
            t_in_standard = time.time()
            while (t < t_max):
                n_ent = negat_ent(N,Lind_matr,t)
                if np.abs(np.real(n_ent)) < 1e-13:
                    # print(f'Stand neg = {n_ent}')
                    # t_ent_standard.append(t)
                    dict_data['t_standard'].append(t)
                    print(f'N = {N}, iter = {j}, t_ent_standard = {t}')
                    break
                # else:
                #     neg_old = n_ent
                t = t + Dt
            # print(f'Standard neg old = {neg_old}')   
            t_fin_standard = time.time()
            elapsed_time_standard = elapsed_time_standard + (t_fin_standard - t_in_standard)

        ########################################################################################

    # Save the t_ent data in an Excel file.
    df_data = pd.concat([pd.Series(dict_data[key], name = f'{key}')\
                        for key in dict_data.keys()], axis = 1)
    dir = os.getcwd()
    writer = pd.ExcelWriter(f'{dir}\\ALME_data\\t_ent_N_{N}_{iterations}_iterations_Dt={Dt}_II.xlsx')
    df_data.to_excel(writer, 'DataFrame')
    writer.close()
    print(f'DATA FILE CORRECTLY SAVED FOR {iterations} ITERATIONS - N = {N}')
    

if __name__== '__main__':
    t_in = time.time()
    processes = []
    N_list = [6,7]
    for N in N_list:
        process = multiprocessing.Process(target=ALME_savedata, 
                                          args=(N,))
        processes.append(process)
        process.start()
    for proc in processes:
        proc.join()
    print(f'elapsed time = {time.time() - t_in}')

