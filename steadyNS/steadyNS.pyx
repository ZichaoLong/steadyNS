# distutils: language = c++
# cython: language_level=3

import numpy as np
import scipy as sp
import scipy.sparse
from . import QuadPts
from . import poisson

def StiffMat(d,M,N,NE,B,e,E,eMeasure):
    D = (d+1)*(d+2)//2
    C_NUM = M*D*d
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
    W1,Lambda1 = QuadPts.quadpts(d,1)
    cdef double[::1] W1_memview = W1
    cdef double[::1] Lambda1_memview = Lambda1.reshape(-1)
    idx = _StiffMatOO(d, M, N, NE,
            &B_memview[0], &e_memview[0],
            &E_memview[0], &eMeasure_memview[0],
            W1.size, &W1_memview[0], &Lambda1_memview[0], 
            &I_memview[0], &J_memview[0], &data_memview[0])
    print("idx=",idx,", C_NUM=",C_NUM)
    C = []
    for l in range(d):
        C.append(sp.sparse.coo_matrix((data[l::d],(I[l::d],J[l::d])),shape=[M,N+NE]))
    for l in range(d):
        C[l] = C[l].tocsr()
    return C

def ReturnU(d,N,NE,B):
    U = np.zeros((d,N+NE))
    U[0,B==4] = -1
    return U

def EmbedU(d,N,NE,B,U0):
    U = np.zeros((d,N+NE))
    U[0,B==4] = -1
    for l in range(d):
        U[l,B==0] = U0[l]
    return U

def RHI(U0,d,M,N,NE,B,e,E,eMeasure):
    B = np.ascontiguousarray(B)
    e = np.ascontiguousarray(e)
    cdef int[::1] B_memview = B
    cdef int[::1] e_memview = e.reshape(-1)
    E = np.ascontiguousarray(E)
    eMeasure = np.ascontiguousarray(eMeasure)
    cdef double[::1] E_memview = E.reshape(-1)
    cdef double[::1] eMeasure_memview = eMeasure
    W5,Lambda5 = QuadPts.quadpts(d,5)
    cdef double[::1] W5_memview = W5
    cdef double[::1] Lambda5_memview = Lambda5.reshape(-1)
    U = EmbedU(d,N,NE,B,U0)
    cdef double[::1] U_memview = U.reshape(-1)
    rhi = np.empty_like(U)
    cdef double[::1] rhi_memview = rhi.reshape(-1)
    _RHI(d, M, N, NE, &B_memview[0], &e_memview[0], &E_memview[0], 
            &eMeasure_memview[0], W5.size, &W5_memview[0], &Lambda5_memview[0], 
            &U_memview[0], &rhi_memview[0])
    rhi = rhi[:,B==0]
    return rhi

