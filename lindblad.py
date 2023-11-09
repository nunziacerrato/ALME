''' This library implements some basic functions to study the properties of an open quantum system
    and arrive at the computation of the negativity of entanglement.
'''
import numpy as np
import qutip

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

def Dissipator(N,RM_D,matrix):
    r''' Function that creates the dissipator as a superoperator acting on the input matrix, starting
        from the Kossakowski matrix constructed from a positive matrix, with unit trace, given as input.
        Here it is ensured that the trace of the Kossakowski matrix is equal to :math:`N`.
        
        Parameters
        ----------
        N : int
            Dimension of the input matrix.
        RM_D : ndarray
                Positive matrix with unit trace.
                This matrix can be sampled from the Ginibre ensemble using the QuTip library in the following way:
                RM_D = np.array(qutip.rand_dm_ginibre(:math:`(N^2-1)`, rank=None)).
        matrix : ndarray
                The input matrix.
            
        Returns
        -------
        out : ndarray
            Input matrix after the action of the dissipator.
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

    # Finally the dissipator is built
    Diss = np.zeros((N,N), dtype=complex)
    for j in range(N**2 -1):
        Diss = Diss + L[j]@matrix@L_dagg[j] - 0.5*(L_dagg[j]@L[j]@matrix + matrix@L_dagg[j]@L[j])

    return Diss

def Hamiltonian_part(N,RM_H,matrix):
    r''' Function that builds the unitary Hamiltonian contribution to the Lindbladian, given by the
        commutator between a Hamiltonian and an input matrix.
        :math:`\hbar = 1` required to prevent overflow.
        
        Parameters
        ----------
        N : int
            Dimension of the input matrix.
        RM_H : ndarray
                Hamiltonian matrix.
                This matrix can be sampled from the GUE ensemble using the TeNPy library in the following way:
                RM_H = tenpy.linalg.random_matrix.GUE(:math:`(N,N)`).
        matrix : ndarray
                 The input matrix.
        Returns
        -------
        out : ndarray
              Input matrix after the action of the Hamiltonian contribution to the Lindbladian.
    '''
    Hamilt_part = (-1j)*(RM_H@matrix - matrix@RM_H)

    return Hamilt_part

def Lindbladian(N,RM_D,RM_H,matrix,alpha,gamma):
    r''' Function that builds the Lindbladian applied to an input state by adding the unitary
        Hamiltonian contribution and the dissipator.
        
        Parameters
        ----------
        
        N : int
            Dimension of the input matrix.
        RM_D : ndarray
                Matrix used to construct the dissipator.
                This matrix can be sampled from the Ginibre ensemble using the QuTip library in the following way:
                RM_D = np.array(qutip.rand_dm_ginibre(:math:`(N^2-1)`, rank=None)).
        RM_H : ndarray
                Hamiltonian matrix.
                This matrix can be sampled from the GUE ensemble using the TeNPy library in the following way:
                RM_H = tenpy.linalg.random_matrix.GUE(:math:`(N,N)`).
        matrix : ndarray
                Input matrix.
        alpha : float
                Parameter that regulates the strenght of the unitary Hamiltonian contribution to the Lindbladian.
        gamma : float
                Parameter that regulates the strenght of the dissipator.
        
        Returns
        -------
        out : ndarray
            Input matrix after the action of the Lindbladian.
    '''
    L = alpha*Hamiltonian_part(N,RM_H,matrix) + gamma*Dissipator(N,RM_D,matrix)

    return L

def Lindbladian_matrix(N,RM_D,RM_H,alpha,gamma):
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
            A = FF[m]@Lindbladian(N,RM_D,RM_H,FF[n],alpha,gamma)
            lindbladian_matr[m,n] = np.trace(A)

    return lindbladian_matr

def phi_t(N,Lind_matr,t):
    r''' Function that builds the CPT channel :math:`\Phi(t)` associated with an input Lindbladian :math:`\mathcal{L}`
        as :math:`\Phi(t) = e^{(\mathcal{L} t)}`.

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
                Matrix of dimension :math:`(N^2 \times N^2)` which represents the CPT channel
                associated to the Lindbladian matrix.
    '''
    LL = Lind_matr
    lind_eigval, lind_eigvect = np.linalg.eig(LL)

    mat_exp = np.diag(np.exp(lind_eigval*t))

    phi_t = lind_eigvect@mat_exp@np.linalg.inv(lind_eigvect)

    return phi_t

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

def choi_st(N,Lind_matr,t):
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

    phi_t_HS = phi_t(N,Lind_matr,t)

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

def choi_transp(N,Lind_matr,t):
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
    choi_state = qutip.Qobj(choi_st(N,Lind_matr,t), dims = [[N,N],[N,N]], shape = (N**2,N**2))
    choi_state_transpose_B = qutip.partial_transpose(choi_state, [0,1])

    return np.array(choi_state_transpose_B)

def negat_ent(N,Lind_matr,t):
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
    choi_trans_eigval = np.linalg.eigvalsh(choi_transp(N,Lind_matr,t))
    neg_ent=0
    for i in range(N**2):
        neg_ent = neg_ent + np.absolute(choi_trans_eigval[i]) - choi_trans_eigval[i]

    return neg_ent/2
