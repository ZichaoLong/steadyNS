# distutils: language = c++
# cython: language_level=3

import numpy as np
import scipy as sp
import scipy.sparse
from . import QuadPts

def Poisson_countStiffMatData(d,M,N,NE,B,e):
    B = np.ascontiguousarray(B)
    e = np.ascontiguousarray(e)
    cdef int[::1] B_memview = B
    cdef int[::1] e_memview = e.reshape(-1)
    return _Poisson_countStiffMatData(d, M, N, NE, 
            &B_memview[0], &e_memview[0])

def Poisson_StiffMat(C_NUM,d,nu,M,N,NE,B,e,E,eMeasure):
    I = np.empty(C_NUM,dtype=np.int32)
    J = np.empty(C_NUM,dtype=np.int32)
    data = np.empty(C_NUM,dtype=float)
    cdef int[::1] I_memview = I;
    cdef int[::1] J_memview = J;
    cdef double[::1] data_memview = data;
    B = np.ascontiguousarray(B)
    e = np.ascontiguousarray(e)
    cdef int[::1] B_memview = B
    cdef int[::1] e_memview = e.reshape(-1)
    E = np.ascontiguousarray(E)
    eMeasure = np.ascontiguousarray(eMeasure)
    cdef double[::1] E_memview = E.reshape(-1)
    cdef double[::1] eMeasure_memview = eMeasure
    W2,Lambda2 = QuadPts.quadpts(d,2)
    cdef double[::1] W2_memview = W2
    cdef double[::1] Lambda2_memview = Lambda2.reshape(-1)
    idx = _Poisson_StiffMatOO(C_NUM, d, nu, M, N, NE,
            &B_memview[0], &e_memview[0],
            &E_memview[0], &eMeasure_memview[0],
            W2.size, &W2_memview[0], &Lambda2_memview[0], 
            &I_memview[0], &J_memview[0], &data_memview[0])
    print("idx=",idx,", C_NUM=",C_NUM)
    C = sp.sparse.coo_matrix((data,(I,J)),shape=[N+NE,N+NE])
    print("convert C to csr sparse matrix")
    C = C.tocsr()
    return C

def ReturnU(N,NE,B):
    U = np.zeros(N+NE)
    U[B==4] = -1
    return U

def EmbedU(N,NE,B,U0):
    U = np.zeros(N+NE)
    U[B==4] = -1
    U[B==0] = U0
    return U

