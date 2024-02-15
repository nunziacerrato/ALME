''' This file serves a library. It contains all the basic functions to obtain an approximate version
    of the Lindblad master equation and compute the evolved quantum state after the interction with
    the environment. The approach used is based on the following two works:
    - Steinbach, J. and Garraway, B. M. and Knight, P. L., High-order unraveling of
      master equations for dissipative evolution, Phys. Rev. A, 51(4), 3302, 1995.
    - Yu Cao and Jianfeng Lu, Structure-preserving numerical schemes for Lindblad
      equations, arxiv 2103.01194, 2021.
'''

import os
import time
import numpy as np
import qutip
import tenpy
import scipy.linalg
import pandas as pd


def E_matr_base_can(N,i,j):
    r''' Function that constructs a matrix of the canonical basis of matrices in dimension :math:`N`,
        with a :math:`1` in position :math:`(i, j)` and :math:`0` otherwise.

        Parameters
        ----------
        N : int
            Dimension of the matrix.
        i : int
            Row index.
        j : int
            Column index.

        Returns
        -------
        out : ndarray
            Matrix of the computational basis of dim. :math:`N` with a :math:`1` in the position :math:`(i,j)`.
    '''
    E = np.zeros((N,N))
    E[i][j] = 1
    return E

def F_matr_base_hs(N):
    r''' Function that creates an array of dimension :math:`(N^2 \times N \times N)` made up of :math:`N^2` matrices of dimension
        :math:`(N \times N)` which form an orthonormal basis with respect to the Hilbert-Schmidt scalar product.
        These matrices are given by: :math:`\mathbb{1}(N)` and the infinitesimal generators of :math:`SU(N)`.
        
        Parameters
        ----------
        N : int
            Dimension of the matrix.
        
        Returns
        -------
        out : ndarray
            Array of dimension :math:`(N^2 \times N \times N)` made up of :math:`N^2` base matrices,
            orthonormal with respect to the Hilbert-Schmidt scalar product.
    '''
    F_base = np.zeros((N**2,N,N), dtype=complex)
    F_base[0] = (1/np.sqrt(N))*np.eye(N)

    # Symmetric (S) and antisymmetric (A) matrices are created - there are N (N-1) / 2 of each type
    elem = 1
    for m in range(N):
        for k in range(N):
            if k>m:
                F_base[elem] = (E_matr_base_can(N,m,k) + E_matr_base_can(N,k,m))*(1/np.sqrt(2))
                elem += 1

                F_base[elem] = (E_matr_base_can(N,m,k) - E_matr_base_can(N,k,m))*((-1j)/np.sqrt(2))
                elem += 1

    # Diagonal (D) matrices are created - there are (N-1)
    sum_k = 0
    for l in range(1,N):
        for kk in range(1,l+1):
            sum_k = sum_k + E_matr_base_can(N,kk-1,kk-1)
        F_base[elem] = (1/np.sqrt(l*(l+1)))*(sum_k - (l)*(E_matr_base_can(N,l,l)))
        sum_k = 0
        elem += 1

    return F_base

def matrix_to_HS(N,matrix):
    r''' Function that, given an input matrix, returns the vector which represents 
        the input matrix with respect to the Hilbert-Schmidt basis.
        
        Parameters
        ----------
        N : int
            Dimension of the input matrix.
        matrix : ndarray
            Input matrix.
        
        Returns
        -------
        out : 1D array
            The vector which represents the input matrix with respect to
            the Hilbert-Schmidt basis.
    '''
    FF_HS = F_matr_base_hs(N)
    coeff_vect = np.zeros(N**2)

    for item in range(N**2):
        coeff_vect[item] = np.real(np.trace(matrix@FF_HS[item]))

    return coeff_vect

def max_ent(N):
    r''' Function that constructs the density matrix associated to the maximally entangled state
        :math:`\ket{\Psi}_{AB} = \frac{1}{\sqrt{N}}\sum_{k=1}^{N} \ket{k}_{A} \otimes \ket{k}_{B}`,
        where :math:`N` is the dimension of the Hilbert space.
        
        Parameters
        ----------
        N : int
            Dimension of the Hilbert space.
        
        Returns
        -------
        out : ndarray
                Matrix of dimension :math:`(N^2 \times N^2)` which represents the density matrix of 
                the maximally entangled state 
                :math:`\ket{\Psi}_{AB} = \frac{1}{\sqrt{N}}\sum_{k=1}^{N} \ket{k}_{A} \otimes \ket{k}_{B}`.
    '''
    summ_outer = np.zeros((N**2,N**2))

    for i in range(N):
        for j in range(N):
            summ_outer = summ_outer + np.outer(E_matr_base_can(N, i, i),E_matr_base_can(N, j, j))
    summ_outer = summ_outer/N

    return summ_outer




def create_L_L_dagg(N,eigvect_K, eigval_K):
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
    state_trans_eigval = np.linalg.eigvalsh(state_transpose_B)
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




def Dissipative_term(eigvect_K, eigval_K, N, input_state):
    r''' Function that creates the dissipator as a superoperator acting on the input matrix, starting
        from the Kossakowski matrix constructed from a positive matrix, with unit trace, given as input.
        Here it is ensured that the trace of the Kossakowski matrix is equal to :math:`N`.
        
        Parameters
        ----------
        eigvect_K : ndarray
            Matrix of the eigenvectors of the Kossakoswki matrix.
        eigval_K : ndarray
            Vector of the eigenvalues of the Kossakowski matrix.
        N : int
            Dimension of the input density matrix.
        input_state : ndarray
            The input density matrix.
            
        Returns
        -------
        out : ndarray
            Input density matrix after the action of the dissipator.
    '''

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

    # Finally the dissipator is built
    Diss = np.zeros((N,N), dtype=complex)
    for j in range(N**2 -1):
        Diss = Diss + L[j]@input_state@L_dagg[j] - 0.5*(L_dagg[j]@L[j]@input_state + input_state@L_dagg[j]@L[j])

    return Diss

def Hamiltonian_term(H, input_state):
    r''' Function that builds the unitary Hamiltonian contribution to the Lindbladian, given by the
        commutator between a Hamiltonian and an input matrix.
        :math:`\hbar = 1` required to prevent overflow.
        
        Parameters
        ----------
        N : int
            Dimension of the input matrix.
        H : ndarray
            Hamiltonian matrix.
            This matrix can be sampled from the GUE ensemble using the TeNPy library in the following way:
            RM_H = tenpy.linalg.random_matrix.GUE(:math:`(N,N)`).
        input_state : ndarray
            The input matrix.
        Returns
        -------
        out : ndarray
            Input matrix after the action of the Hamiltonian contribution to the Lindbladian.
    '''
    Hamilt_part = (-1j)*(H@input_state - input_state@H)

    return Hamilt_part

def Lindbladian(H, eigvect_K, eigval_K, N, input_state, alpha = 1, gamma = 1):
    r''' Function that construct the Lindblad superoperator applied to an input state by adding the
    Hamiltonian term and the Dissipator.
        
    Parameters
    ----------
    H : ndarray
        Hamiltonian matrix associated to the Markovian noise acting on one subsystem
    eigvect_K : ndarray
        Matrix of the eigenvectors of the Kossakoswki matrix.
    eigval_K : ndarray
        Vector of the eigenvalues of the Kossakowski matrix.
    N : int
        Dimension of the Hilbert space of one of the two subsystems forming the maximally 
        entangled state.
    input_state : ndarray
        The input density matrix.
    alpha : float
        Parameter representing the strenght of the Hamiltonian contribution to the Lindbladian.
        Default value equal to 1.
    gamma : float
        Parameter representing the strenght of the dissipative contribution to the Lindbladian.
        Default value equal to 1.

    Returns
    -------
    out : ndarray
        Density matrix after the action of the Lindbladian.
    '''
    L = alpha*Hamiltonian_term(H, input_state) \
        + gamma*Dissipative_term(eigvect_K, eigval_K, N, input_state)

    return L

def Lindbladian_matrix(H, eigvect_K, eigval_K, N, alpha = 1, gamma = 1):
    r''' Function that computes the matrix associated with the Lindblad superoperator written with
        respect to the Hilbert-Schmidt matrix basis. Called :math:`F[m]` these matrices, for :math:`m = 1,\dots,N^2`,
        the elements of the Lindbladian matrix are: :math:`\hat{\mathcal{L}}[m,n]=Tr(F[m]\mathcal{L}(F[n]))`.
        
        Parameters
        ----------
        N : int
            Dimension of the Hilbert space.
        RM_D : ndarray
                Matrix used to construct the dissipator.
                This matrix can be sampled from the Ginibre ensemble using the QuTip library in the following way:
                RM_D = np.array(qutip.rand_dm_ginibre(:math:`(N^2-1)`, rank=None)).
        RM_H : ndarray
                Hamiltonian matrix.
                This matrix can be sampled from the GUE ensemble using the TeNPy library in the following way:
                RM_H = tenpy.linalg.random_matrix.GUE(:math:`(N,N)`).
        alpha : float
                Parameter that regulates the strenght of the unitary Hamiltonian contribution to the Lindbladian.
        gamma : float
                Parameter that regulates the strenght of the dissipator.
        
        Returns
        -------
        out : ndarray
                Lindbladian matrix of dimension :math:`(N^2 \times N^2)`, written in the Hilbert-Schmidt
                matrix basis.
    '''
    FF = F_matr_base_hs(N)
    lindbladian_matr = np.zeros((N**2,N**2), dtype=complex)

    for m in range(N**2):
        for n in range(N**2):
            A = FF[m]@Lindbladian(H, eigvect_K, eigval_K, N, FF[n], alpha = alpha, gamma = gamma)
            lindbladian_matr[m,n] = np.trace(A)

    return lindbladian_matr

def phi_t(lind_eigval, lind_eigvect, lind_eigvect_inv, t):
    r''' Function that construct the LCPTP channel :math:`\Phi(t)` associated with the Lindbladian
        :math:`\mathcal{L}` as :math:`\Phi(t) = e^{(\mathcal{L} t)}`, starting from the eigenvalues
        and the eigenvector of the Lindbladian, passed from the outside.
        Note that this choice prevents diagonalizing the Lindbladian at each time step.

        Parameters
        ----------
        lind_eigval : ndarray
            Vector of the eigenvalues of the Lindbladian
        lind_eigvect : ndarray
            Matrix of the eigenvectors of the Lindbladian
        t : float
            Time at which the :math:`\Phi(t)` superoperator is evaluated.
        
        Returns
        -------
        out : ndarray
            Matrix which represents the LCPTP channel associated with the Lindbladian matrix.
    '''

    mat_exp = np.diag(np.exp(lind_eigval*t))
    # phi_t = lind_eigvect@mat_exp@np.linalg.inv(lind_eigvect)
    phi_t = lind_eigvect@mat_exp@lind_eigvect_inv
    return phi_t

def choi_st(N,lind_eigval, lind_eigvect, lind_eigvect_inv, t):
    r''' Function that computes the Choi-state associated to the CPT channel :math:`\Phi(t)`.
        
        Parameters
        ----------
        N : int
            Dimension of the Hilbert space.
        Lind_matr : ndarray
                    Lindbladian matrix of dimension :math:`(N^2 \times N^2)`.
        t : float
            Time at which the :math:`\Phi(t)` operator is evaluated.
        
        Returns
        -------
        out : ndarray
            Matrix of dimension :math:`(N^2 \times N^2)` which represents the Choi-state associated to
            the CPT channel :math:`\Phi(t)` obtained from the Lindbladian matrix in input.
    '''
    phi_t_HS = np.zeros((N**2,N**2), dtype=complex)

    phi_t_HS = phi_t(lind_eigval, lind_eigvect, lind_eigvect_inv, t)

    # Build the maximally entangled state and write the coefficient vector that uniquely identifies
    # it in the Hilbert-Schmidt base.
    max_ent_state = max_ent(N)

    FF = np.kron(F_matr_base_hs(N),F_matr_base_hs(N))
    coeff = np.zeros((N**4))
    for i in range(N**4):
        coeff[i] = np.real(np.trace(max_ent_state@FF[i]))

    # Construct the extended channel (Phi(t) x Id(N**2)), which will be written in the HS base, and
    # apply it to the coefficient vector that identifies the maximally entangled state in the
    # Hilbert-Schmidt base.
    ext_chann = np.kron(phi_t_HS,np.eye(N**2))

    choi_state_coeff = ext_chann@coeff

    # Reconstruct the Choi-state from the previous output vector
    choi_state = np.zeros((N**2,N**2))
    for ii in range(N**4):
        el = choi_state_coeff[ii]*FF[ii]
        choi_state = choi_state + el

    return choi_state

def choi_transp(N,lind_eigval, lind_eigvect, lind_eigvect_inv, t):
    r''' Function that computes the partial transpose of the Choi-state with respect to the system B.
        
        Parameters
        ----------
        N : int
            Dimension of the Hilbert space.
        Lind_matr : ndarray
                    Lindbladian matrix of dimension :math:`(N^2 \times N^2)`.
        t : float
            Time at which the :math:`\Phi(t)` operator is evaluated.
        
        Returns
        -------
        out : ndarray
                Matrix of dimension :math:`(N^2 \times N^2)` which represents the partial transpose
                of the Choi-state with respect to the system B.
    '''
    choi_state = qutip.Qobj(choi_st(N,lind_eigval, lind_eigvect, lind_eigvect_inv, t), dims = [[N,N],[N,N]], shape = (N**2,N**2))
    choi_state_transpose_B = qutip.partial_transpose(choi_state, [0,1])

    return np.array(choi_state_transpose_B)

def negat_ent(N,lind_eigval, lind_eigvect, lind_eigvect_inv, t):
    r''' Function that computes the negativity of entanglement, defined as 
        :math:`\mathcal{N}(\rho_{C-J}^{(\hat{\phi_{t}})}) = \sum_{i} \frac{\vert \lambda_{i} \vert - \lambda_{i}}{2}`, 
        where :math:`\lambda_{i}` are the eigenvalues of the partial transpose of the Choi-state.

        Parameters
        ----------
        N : int
            Dimension of the Hilbert space.
        Lind_matr : ndarray
                    Lindbladian matrix of dimension :math:`(N^2 \times N^2)`.
        t : float
            Time at which the :math:`\Phi(t)` operator is evaluated.
        
        Returns
        -------
        out : float
                The negativity of entanglement.
    '''
    choi_trans_eigval = np.linalg.eigvalsh(choi_transp(N,lind_eigval, lind_eigvect, lind_eigvect_inv, t))
    neg_ent=0
    for i in range(N**2):
        neg_ent = neg_ent + np.absolute(choi_trans_eigval[i]) - choi_trans_eigval[i]

    return neg_ent/2


if __name__ == '__main__':
    
    t0, Dt = 0, 0.001
    iterations = 355

    Kraus = True
    Standard = False
    save_time_N = False
    alpha, gamma = 1, 1

    N_list = [8]

    dict_time_N = {'N' : [], 'elaps_approx_Kraus' : [], 'elaps_stand' : []}

    for N in N_list:
        dict_time_N['N'].append(N)
        dict_data = {'t_approx_Kraus' : [], 't_standard' : []}

        elapsed_time_approx_Kraus = 0
        elapsed_time_standard = 0

        for j in range(iterations):

            # Sample the Hamiltonian and sample and diagonalize the Kossakowski matrix associated
            #  with noise acting locally on the subsystem S
            ham = tenpy.linalg.random_matrix.GUE((N,N))
            RM_D = np.array(qutip.rand_dm_ginibre(N**2 -1, rank=None))
            K = N*RM_D
            eigval_K, eigvect_K = np.linalg.eigh(K)

            # Construct and diagonalize the Lindbladian matrix associated with the Markovian noise
            LL = Lindbladian_matrix(ham, eigvect_K, eigval_K, N, alpha = 1, gamma = 1)
            lind_eigval, lind_eigvect = np.linalg.eig(LL)
            lind_eigvect_inv = np.linalg.inv(lind_eigvect)
                    
        
            L, L_dagg = create_L_L_dagg(N,eigvect_K, eigval_K)
            L_new = np.zeros((N**2-1, N**2, N**2), dtype=complex)
            L_new_dagg = np.zeros((N**2-1, N**2, N**2), dtype=complex)
            for i in range(N**2-1):
                L_new[i] = np.kron(L[i], np.eye(N))
                L_new_dagg[i] = np.kron((np.conjugate(L[i])).T, np.eye(N))
            
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

                t_in_standard = time.time()
                while(neg >= 1e-13):

                    neg = negat_ent(N,lind_eigval, lind_eigvect, lind_eigvect_inv, t)
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
        writer = pd.ExcelWriter(f'{dir}\\ALME_data\\3_t_ent_N_{N}_{iterations}_iterations_Dt={Dt}.xlsx')
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
                pd.ExcelWriter(f'{dir}\\ALME_data\\33_times_for_N_t_ent_iters={iterations}_Dt={Dt}.xlsx')
        df_time_N.to_excel(writer_times_N, 'DataFrame')
        writer_times_N.close()
        print(f'TIME N FILE CORRECTLY SAVED')







