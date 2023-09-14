''' This file serves a library. It contains all the basic functions to obtain an approximate version
    of the Lindblad master equation and compute the evolved quantum state after the interction with
    the environment. The approach used is based on the following two works:
    - Steinbach, J. and Garraway, B. M. and Knight, P. L., High-order unraveling of
      master equations for dissipative evolution, Phys. Rev. A, 51(4), 3302, 1995.
    - Yu Cao and Jianfeng Lu, Structure-preserving numerical schemes for Lindblad
      equations, arxiv 2103.01194, 2021.
'''

import numpy as np
import qutip
import scipy.linalg
from lindblad import *

def create_L_L_dagg(N,RM_D):
  ''' This function creates the Lindblad operators starting from a given Kossakowski matrix.
      
      Parameters
      ----------
      N : int
        Dimension of the system of interest
      RM_D : ndarray
        Kossakowsky matrix: positive matrix with unit trace.
        This matrix can be sampled from the Ginibre ensemble using the QuTip library as follows:
        RM_D = np.array(qutip.rand_dm_ginibre(:math:`(N^2-1)`, rank=None))
      
      Results
      -------
      L : ndarray
        Array of dimension :math:`(N^2 \times N \times N)` of :math:`(N^2)` Lindblad operators
      L_dagg : ndarray
        Conjugate transpose of the array L
  '''

  K = N*RM_D

  # Diagonalize the normalized Wishart matrix -> Kossakowski matrix
  eigval_K, eigvect_K = np.linalg.eigh(K)

  # Build Lindblad operators as an array of three indices: N*2 - 1 operators of dimension (N x N)
  F = F_matr_base_hs(N)
  L = np.zeros((N**2 -1,N,N), dtype=complex)
  L_dagg = np.zeros((N**2 -1,N,N), dtype=complex)

  for k in range(N**2 -1):
      l = np.zeros((N,N), dtype=complex)
      for m in range(N**2 -1):
          l = l + eigvect_K[m,k]*F[m+1]  # You have to exclude the first element of F, Id(N).
      l = l*np.sqrt(eigval_K[k])
      L[k] = l
      L_dagg[k] = (np.conjugate(l)).T
  
  return L, L_dagg

def kraus(oper,state):
    ''' This function computes the Kraus superoperator associated with the input operator (oper) and
        gives as output the evolved input state.
        Parameters
        ----------
        oper : ndarray
            Input operator
        state : ndarray
            Input quantum state

        Results
        -------
        result : ndarray
            Evolved quantum state after the action of the input operator
    '''
    result = oper @ state @ (np.conjugate(oper)).T
    return result

def H_eff(H,lind):
    ''' This function computes the effective Hamiltonian as defined in
        "Phys. Rev. A, 51(4), 3302, 1995".
        
        Parameters
        ----------
        H : ndarray
            Hamiltonian of the system of interest
        lind : ndarray
            Lindblad operators associated to the Markovian noise

        Results
        -------
        result : ndarray
            Effective Hamiltonian
    '''
    num_lind = lind.shape[0]
    NN = lind.shape[1]
    sum_lind = np.zeros((NN,NN), dtype=complex)
    for k in range(num_lind):
        sum_lind += (np.conjugate(lind[k])).T @ lind[k]
    result = H + (1/2j)*sum_lind
    return result

def Lind_J(Heff,state):
    ''' This function computes the term L_J of the Lindbladian 
        as reported in "Phys. Rev. A, 51(4), 3302, 1995".
    
        Parameters
        ---------
        Heff : ndarray
            Effective Hamiltonian of the system
        state : ndarray
            Input quantum state

        Results
        -------
        result : ndarray
            Output state after the action of the term L_J
    '''
    J = -1j*Heff
    result = J @ state + state @ (np.conjugate(J)).T
    return result

def Lind_L(lind,state):
    ''' This function computes the term L_L of the Lindbladian
        as reported in "Phys. Rev. A, 51(4), 3302, 1995".
        
        Parameters
        ----------
        lind : ndarray
            Lindblad operators associated to the Markovian noise
        state : ndarray
            Input quantum state

        Results
        -------
        result : ndarray
            Output state after the action of the term L_L
    
    '''
    num_lind = lind.shape[0]
    N = state.shape[0]
    result = np.zeros((N,N), dtype=complex)
    for k in range(num_lind):
        result += lind[k] @ state @ (np.conjugate(lind[k])).T
    return result

def neg_ent(state,N):
    ''' This function computes the negativity of entanglement of the input state.

        Parameters
        ----------
        N : int
            Dimension of the subsystem of the joint system of interest.
        state : ndarray
            Input state
        
        Returns
        -------
        result : float
            The negativity of entanglement.
    '''
    state_qutip = qutip.Qobj(state, dims = [[N,N],[N,N]], shape = (N**2,N**2))
    state_transpose_B = np.array(qutip.partial_transpose(state_qutip, [0,1]))
    state_trans_eigval = np.linalg.eigvals(state_transpose_B)
    neg = 0
    for i in range(N**2):
        neg = neg + np.absolute(state_trans_eigval[i]) - state_trans_eigval[i]
    return neg/2

def MP_IIord(H,lind,Dt,state):
    ''' This function computes the (approximated) quantum state which is the ouput of the 
        approximated Lindblad master equation. This is a second order approximation
        from "Yu Cao and Jianfeng Lu, Structure-preserving numerical schemes for Lindblad
        equations, arxiv 2103.01194, 2021."
      
        Parameters
        ----------
        H : ndarray
            Hamiltonian of the system of interest
        lind : ndarray
            Lindblad operators associated to the Markovian noise
        Dt : float
            Time step
        state : ndarray
            Input quantum state

        Returns
        -------
        result : ndarray
            Approximated state of the system of interest after the interaction with the
            (Markovian) environment

    '''

    d = state.shape[0]
    Heff = H_eff(H,lind)
    kraus_I_arg = np.eye(d) + (-1j*Dt)*Heff + ((-1j*Dt)**2)*(Heff@Heff)/2
    I_state = kraus(kraus_I_arg,state)

    kraus_II_arg = np.eye(d) + (-1j*Dt/2)*Heff
    state_after_kraus = kraus(kraus_II_arg,state)
    state_after_lindL = Lind_L(lind,state_after_kraus)
    state_after_second_kraus = kraus(kraus_II_arg,state_after_lindL)
    II_state = Dt*state_after_second_kraus

    I_LindL = Lind_L(lind,state)
    II_LindL = Lind_L(lind,I_LindL)
    III_state = ((Dt**2)/2)*II_LindL

    result = I_state + II_state + III_state
    
    return result

def unraveling_LME_II(state,H,lind,lind_dagg,dt,N):
    ''' This function computes the (approximated) quantum state which is the ouput of an 
        approximated Lindblad master equation. The output state is be approximated with an error 
        of :math:`O(dt^3)`, following the approximation scheme reported in 
        "Phys Rev A, 51(4), 3302, 1995", Eq. (7).

    Parameters
    ----------
    state : nddarray
        Input quantum state
    H : ndarray
        Hamiltonian of the system
    lind : ndarray
        Lindblad operators acting on the system
    lind_dagg : ndarray
        Dagger of the Lindblad operators acting on the system
    dt : float
        Time step
    N : int
        Dimension of one subsystem of the system of interest

    Returns
    -------
    state : ndarray
        Approximated state of the system of interest after the interaction with the
        (Markovian) environment
    '''
    
    d = state.shape[0]
    Heff = (-1j*dt)*(H_eff(H,lind))

    U = scipy.linalg.expm(Heff)
    U_dagg = (np.conjugate(U)).T

    I_term = U @ state @ U_dagg

    II_term = np.zeros((d,d), dtype=complex)
    III_term = np.zeros((d,d), dtype=complex)
    IV_term = np.zeros((d,d), dtype=complex)
    
    for j in range(N**2-1):
        II_term = II_term + (0.5 * dt)* U @ lind[j] @ state @ lind_dagg[j] @ U_dagg
        III_term = III_term + (0.5 * dt) * lind[j] @ U @ state @ U_dagg @ lind_dagg[j]

    for i in range(N**2-1):
        for j in range(N**2-1):
            IV_term = IV_term + (0.5 * dt * dt) * U @ lind[i] @ lind[j] @ state @ lind_dagg[j] @ lind_dagg[i] @ U_dagg

    state = I_term +  II_term + III_term + IV_term
    state_trace = np.trace(state)

    return state/state_trace


