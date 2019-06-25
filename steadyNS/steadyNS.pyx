# distutils: language = c++
# cython: language_level=3

import numpy as np
import scipy as sp
import scipy.sparse
from . import QuadPts

def countStiffMatData(d,M,N,NE,B,P,e):
    B = np.ascontiguousarray(B)
    P = np.ascontiguousarray(P)
    e = np.ascontiguousarray(e)
    cdef int[::1] B_memview = B
    cdef int[::1] P_memview = P
    cdef int[::1] e_memview = e.reshape(-1)
    return _countStiffMatData(d, M, N, NE, 
            &B_memview[0], &P_memview[0], &e_memview[0])

def StiffMat(C_NUM,d,nu,M,N,NE,B,P,e,E,eMeasure):
    I = np.empty(C_NUM,dtype=np.int32)
    J = np.empty(C_NUM,dtype=np.int32)
    data = np.empty(C_NUM,dtype=float)
    cdef int[::1] I_memview = I;
    cdef int[::1] J_memview = J;
    cdef double[::1] data_memview = data;
    B = np.ascontiguousarray(B)
    P = np.ascontiguousarray(P)
    e = np.ascontiguousarray(e)
    cdef int[::1] B_memview = B
    cdef int[::1] P_memview = P
    cdef int[::1] e_memview = e.reshape(-1)
    E = np.ascontiguousarray(E)
    eMeasure = np.ascontiguousarray(eMeasure)
    cdef double[::1] E_memview = E.reshape(-1)
    cdef double[::1] eMeasure_memview = eMeasure
    W1,Lambda1 = QuadPts.quadpts(d,1)
    W2,Lambda2 = QuadPts.quadpts(d,2)
    cdef double[::1] W1_memview = W1
    cdef double[::1] W2_memview = W2
    cdef double[::1] Lambda1_memview = Lambda1.reshape(-1)
    cdef double[::1] Lambda2_memview = Lambda2.reshape(-1)
    idx = _StiffMatOO(C_NUM, d, nu, M, N, NE,
            &B_memview[0], &P_memview[0], &e_memview[0],
            &E_memview[0], &eMeasure_memview[0],
            W1.size, &W1_memview[0], &Lambda1_memview[0], 
            W2.size, &W2_memview[0], &Lambda2_memview[0], 
            &I_memview[0], &J_memview[0], &data_memview[0])
    print("idx=",idx,", C_NUM=",C_NUM)
    C = sp.sparse.coo_matrix((data,(I,J)),shape=[d*(N+NE)+M,d*(N+NE)+M])
    C = C.tocsr()
    return C

def RHI(UP,d,M,N,NE,B,e,E,eMeasure):
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
    cdef double[::1] UP_memview = UP
    rhi = np.empty_like(UP)
    cdef double[::1] rhi_memview = rhi
    _RHI(d, M, N, NE, &B_memview[0], &e_memview[0], &E_memview[0], 
            &eMeasure_memview[0], W5.size, &W5_memview[0], &Lambda5_memview[0], 
            &UP_memview[0], &rhi_memview[0])
    return rhi

def setUP(UP,B,d,M,N,NE):
    cdef int[::1] B_memview = B
    cdef double[::1] UP_memview = UP
    cdef int i=0,l=0,dd=d;
    for i in range(N+NE):
        for l in range(dd):
            UP_memview[dd*i+l] = 0
        if B_memview[i]==1 or B_memview[i]==2:
            UP_memview[dd*i] = 1;
    for i in range(d*(N+NE),d*(N+NE)+M):
        UP_memview[i] = 0
    return UP
