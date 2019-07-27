# distutils: language = c++
# cython: language_level=3

import numpy as np
import scipy as sp
import scipy.sparse
from . import QuadPts
from . import poisson

def ReturnU(d,N,NE,B):
    U = np.zeros((d,N+NE))
    U[0,B==1] = 1
    U[0,B==2] = 1
    U[0,B==3] = 1
    return U

def EmbedU(d,N,NE,B,U0):
    U = ReturnU(d,N,NE,B)
    for l in range(d):
        U[l,B==0] = U0[l]
    return U

def UGU(U0,d,M,N,NE,e,E,eMeasure):
    e = np.ascontiguousarray(e)
    cdef int[::1] e_memview = e.reshape(-1)
    E = np.ascontiguousarray(E)
    eMeasure = np.ascontiguousarray(eMeasure)
    cdef double[::1] E_memview = E.reshape(-1)
    cdef double[::1] eMeasure_memview = eMeasure
    W5,Lambda5 = QuadPts.quadpts(d,5)
    cdef double[::1] W5_memview = W5
    cdef double[::1] Lambda5_memview = Lambda5.reshape(-1)
    U = np.ascontiguousarray(U0)
    cdef double[::1] U_memview = U.reshape(-1)
    ugu = np.empty_like(U)
    cdef double[::1] ugu_memview = ugu.reshape(-1)
    _UGU(d, M, N, NE, &e_memview[0], &E_memview[0], &eMeasure_memview[0], 
            W5.size, &W5_memview[0], &Lambda5_memview[0], 
            &U_memview[0], &ugu_memview[0])
    return ugu

def sourceF(d,M,N,NE,e,E,eMeasure):
    e = np.ascontiguousarray(e)
    cdef int[::1] e_memview = e.reshape(-1)
    E = np.ascontiguousarray(E)
    eMeasure = np.ascontiguousarray(eMeasure)
    cdef double[::1] E_memview = E.reshape(-1)
    cdef double[::1] eMeasure_memview = eMeasure
    W4,Lambda4 = QuadPts.quadpts(d,4)
    cdef double[::1] W4_memview = W4
    cdef double[::1] Lambda4_memview = Lambda4.reshape(-1)
    D = (d+1)*(d+2)//2
    C_NUM = D*D*M
    I = np.empty(C_NUM,dtype=np.int32)
    J = np.empty(C_NUM,dtype=np.int32)
    data = np.empty(C_NUM,dtype=float)
    cdef int[::1] I_memview = I;
    cdef int[::1] J_memview = J;
    cdef double[::1] data_memview = data;
    idx = _sourceFOO(C_NUM, d, M, N, NE,
            &e_memview[0], &E_memview[0], &eMeasure_memview[0],
            W4.size, &W4_memview[0], &Lambda4_memview[0],
            &I_memview[0], &J_memview[0], &data_memview[0])
    print("idx=",idx,", C_NUM=",C_NUM)
    C = sp.sparse.coo_matrix((data,(I,J)),shape=[N+NE,N+NE])
    C = C.tocsr()
    return C

def QGU(d,M,N,NE,e,E,eMeasure):
    e = np.ascontiguousarray(e)
    cdef int[::1] e_memview = e.reshape(-1)
    E = np.ascontiguousarray(E)
    eMeasure = np.ascontiguousarray(eMeasure)
    cdef double[::1] E_memview = E.reshape(-1)
    cdef double[::1] eMeasure_memview = eMeasure
    W2,Lambda2 = QuadPts.quadpts(d,2)
    cdef double[::1] W2_memview = W2
    cdef double[::1] Lambda2_memview = Lambda2.reshape(-1)
    D = (d+1)*(d+2)//2
    C_NUM = D*(d+1)*d*M
    I = np.empty(C_NUM,dtype=np.int32)
    J = np.empty(C_NUM,dtype=np.int32)
    data = np.empty(C_NUM,dtype=float)
    cdef int[::1] I_memview = I;
    cdef int[::1] J_memview = J;
    cdef double[::1] data_memview = data;
    idx = _QGUOO(C_NUM, d, M, N, NE,
            &e_memview[0], &E_memview[0], &eMeasure_memview[0],
            W2.size, &W2_memview[0], &Lambda2_memview[0],
            &I_memview[0], &J_memview[0], &data_memview[0])
    print("idx=",idx,", C_NUM=",C_NUM)
    C = []
    for l in range(d):
        C.append(sp.sparse.coo_matrix((data[l::d],(I[l::d],J[l::d])),shape=[N,N+NE]))
    for l in range(d):
        C[l] = C[l].tocsr()
    return C

