''' Here we create the plots of the rescaled histograms of PPT times as the dimension N of one
    of the two subsystems varies. The scaling relation is P_N(x) = a_{N}*P_{N_{0}}^{g_{N}}(b*x), 
    where a_{N}, b_{N}, and g_{N} must be determined empirically from the data.
    b_{N} = t_{max}^{N_{0}}/t_{max}^{N}
    a_{N} = P_{N}^{max}/(P_{N_{0}}^{max})^{g_{N}}
    a_{N}/b_{N} \int_{0}^{\infty} (P_{N_{0}}(u))^{g_{N}}du = 1
'''

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy.integrate import quad
from scipy.optimize import fsolve
from scipy.interpolate import interp1d

# Access directory and set basic parameters
path = os.getcwd()
dir = os.path.abspath(os.path.join(path, os.pardir))
N0, Dt, iter = 3, 0.001, 2000
N_list = [3,4,5,6,7,8]
bins = 'auto'

# Read data from the excel file for N = 3
df_Kraus_N0 = pd.read_excel(f'{dir}\\ALME\\ALME_data\\2_t_ent_N_{N0}_{iter}_iterations_Dt={Dt}.xlsx')
t_ent_N0 = df_Kraus_N0['t_approx_Kraus'].dropna().values

# Creare l'istogramma per N0
bin_counts_N0, bin_edges_N0 = np.histogram(t_ent_N0, bins=bins, density=False)  # Istogramma non normalizzato
bin_width_N0 = bin_edges_N0[1] - bin_edges_N0[0]  # Larghezza di un bin

# Trova l'indice corrispondente al valore massimo dell'istogramma
max_index = np.argmax(bin_counts_N0)

# Calcola tau_max_N0 e P_max_N0
tau_max_N0 = (bin_edges_N0[max_index] + bin_edges_N0[max_index + 1]) / 2  # Centro del bin massimo
P_N0_max = bin_counts_N0[max_index] / (len(t_ent_N0) * bin_width_N0)  # Valore normalizzato


# Parametri della distribuzione 3-parameter Gamma per N0 = 3
mu = 1.280 # Traslazione (threshold)
k = 7.122   # Forma (shape)
theta = 0.03602  # Scala (scale)


# Distribuzione Gamma
def P_N0(t):
    if t > mu:
        return ((t - mu)**(k - 1) * np.exp(-(t - mu) / theta)) / (gamma(k) * theta**k)
    else:
        return 0.0  # Per t <= mu, la distribuzione è nulla




# Funzione per calcolare l'integrale per la terza equazione
def integrale(gamma_N):
    integrand = lambda u: P_N0(u)**gamma_N
    result, _ = quad(integrand, 0, np.inf)
    return result

# Funzione per risolvere il sistema per un singolo N
def risolvi_sistema(tau_max_N, P_N_max):
    beta_N = tau_max_N0 / tau_max_N  # Calcolo di beta_N

    # Definizione del sistema di equazioni
    def sistema(variabili):
        alpha_N, gamma_N = variabili
        eq1 = np.log(P_N_max) - np.log(alpha_N) - gamma_N * np.log(P_N0_max)  # Seconda equazione
        eq2 = (alpha_N / beta_N) * integrale(gamma_N) - 1  # Terza equazione
        return [eq1, eq2]

    # Risolvi il sistema
    guess = [0.5, 0.2]  # Valori iniziali
    soluzione = fsolve(sistema, guess)
    alpha_N, gamma_N = soluzione
    return alpha_N, beta_N, gamma_N


results= {'N':[], 'tau_max_N':[], 'P_N_max':[], 'alpha_N':[], 'beta_N':[], 'gamma_N':[]}
for N in N_list:
    results['N'].append(N)
    # Read data from the excel file for N = 3
    df_Kraus_N = pd.read_excel(f'{dir}\\ALME\\ALME_data\\2_t_ent_N_{N}_{iter}_iterations_Dt={Dt}.xlsx')
    # Access to the column 't_ent' and drop NaN values if present
    t_ent_N = df_Kraus_N['t_approx_Kraus'].dropna().values

    # Creare l'istogramma
    bin_counts_N, bin_edges_N = np.histogram(t_ent_N, bins=bins, density=False)  # Istogramma non normalizzato
    bin_width_N = bin_edges_N[1] - bin_edges_N[0]  # Larghezza di un bin

    # Trova l'indice del valore massimo
    max_index = np.argmax(bin_counts_N)

    # Calcola tau_max_N e P_max_N
    tau_max_N = (bin_edges_N[max_index] + bin_edges_N[max_index + 1]) / 2  # Centro del bin massimo
    P_N_max = bin_counts_N[max_index] / (len(t_ent_N) * bin_width_N)  # Valore normalizzato
    results['tau_max_N'].append(tau_max_N)
    results['P_N_max'].append(P_N_max)


    # Itera su tutti i valori di N e calcola i risultati
    alpha_N, beta_N, gamma_N = risolvi_sistema(tau_max_N, P_N_max)
    results['alpha_N'].append(alpha_N)
    results['beta_N'].append(beta_N)
    results['gamma_N'].append(gamma_N)

print(results)





# Read data from the excel file for N = 3
df_Kraus_N0 = pd.read_excel(f'{dir}\\ALME\\ALME_data\\2_t_ent_N_{N0}_{iter}_iterations_Dt={Dt}.xlsx')
# Access to the column 't_ent' and drop NaN values if present
t_ent_N0 = df_Kraus_N0['t_approx_Kraus'].dropna().values


# Costruzione dell'istogramma di P_{N_0}(t)
bin_counts_N0, bin_edges_N0 = np.histogram(t_ent_N0, bins=bins, density=True)  # Istogramma normalizzato
bin_centers_N0 = (bin_edges_N0[:-1] + bin_edges_N0[1:]) / 2  # Centri dei bin
# plt.bar(bin_centers_N0, bin_counts_N0, width=(bin_edges_N0[1] - bin_edges_N0[0]), alpha=0.5, label=r'$P_{N_0}(t)$')

for N in N_list:
    # Read data from the excel file for N = 3
    df_Kraus_N = pd.read_excel(f'{dir}\\ALME\\ALME_data\\2_t_ent_N_{N}_{iter}_iterations_Dt={Dt}.xlsx')
    # Access to the column 't_ent' and drop NaN values if present
    t_ent_N = df_Kraus_N['t_approx_Kraus'].dropna().values

    # Creare l'istogramma
    bin_counts_N, bin_edges_N = np.histogram(t_ent_N, bins=bins, density=True)  # Istogramma normalizzato
    # plt.stairs(bin_counts_N, bin_edges_N, label = f'Original N={N}')
    bin_centers_N = (bin_edges_N[:-1] + bin_edges_N[1:]) / 2  # Centri dei bin
    plt.bar(bin_centers_N, bin_counts_N, width=(bin_edges_N[1] - bin_edges_N[0]), alpha=0.5, label = f'Original N={N}')
    
    index_N = N_list.index(N)
    alpha_N = results['alpha_N'][index_N]
    beta_N = results['beta_N'][index_N]
    gamma_N = results['gamma_N'][index_N]

    # Trasformazione dei tempi e delle altezze per P_N(t)
    # rescaled_centers_t_ent_N = bin_centers_N0/beta_N
    # rescaled_P_N = (alpha_N)*(bin_counts_N0**gamma_N)
    # plt.bar(rescaled_centers_t_ent_N, rescaled_P_N, width=(bin_edges_N[1] - bin_edges_N[0]) / beta_N, alpha=0.5, label=fr'$Rescaled - N={N}$')


    # Creare l'istogramma riscalato
    rescaled_t_ent_N = t_ent_N0/beta_N
    bin_counts_N, bin_edges_N_resc = np.histogram(rescaled_t_ent_N, bins=bins, density=False)  # Istogramma normalizzato
    bin_counts_N_resc = (alpha_N)*(bin_counts_N0**gamma_N)
    plt.stairs(bin_counts_N_resc, bin_edges_N_resc, label=fr'$Rescaled - N={N}$')


    # INTERPOLAZIONE: 
    # Funzione interpolata per P_{N_0}(t)
    P_N0_interpolated = interp1d(bin_centers_N0, bin_counts_N0, kind='cubic', fill_value="extrapolate")
    # Trasformazione dei tempi per P_N(t)
    t_N = bin_centers_N0 / beta_N
    # Trasformazione delle altezze per P_N(t)
    P_N = alpha_N * P_N0_interpolated(bin_centers_N0)**gamma_N
    # Visualizzazione
    # plt.bar(bin_centers_N0, bin_counts_N0, width=(bin_edges_N0[1] - bin_edges_N0[0]), alpha=0.5, label=r'$P_{N_0}(t)$')
    plt.bar(t_N, P_N, width=(bin_edges_N0[1] - bin_edges_N0[0]) / beta_N, alpha=0.5, label=fr'Rescaled_interp - N ={N}$')
    

plt.legend()
plt.show()
