# distutils: language = c++

cdef extern from "steadyNS.h":
    cdef int _reduceP(const int N, int *P)
    cdef int _mergePeriodNodes(const int d, const int M,
            const int *B, const int *P, int *ep)
    cdef int _countStiffMatData(const int d, const int M, const int N,
            const int *B, const int *P, const int *ep)
    cdef int _StiffMatOO(const int C_NUM, const int d, const int M, const int N, const double nu, 
            const int *B, const int *P, const int *ep, const double *Ep, const double *eMeasure, 
            int *I, int *J, double *data)
    cdef int _countPoisson(const int d, const int M, const int N,
            const int *B, const int *P, const int *ep)
    cdef int _PoissonOO(const int C_NUM, const int d, const int M, const int N, const double nu, 
            const int *B, const int *P, const int *ep, const double *Ep, const double *eMeasure, 
            int *I, int *J, double *data)

import numpy as np
import scipy as sp
import scipy.sparse

def reduceP(P):
    P = np.ascontiguousarray(P)
    cdef int[::1] P_memview = P
    _reduceP(P.size,&P_memview[0])
    return P

def mergePeriodicNodes(B, P, e):
    M = e.shape[0]
    d = e.shape[1]-1
    B = np.ascontiguousarray(B)
    P = np.ascontiguousarray(P)
    e = np.ascontiguousarray(e)
    cdef int[::1] B_memview = B
    cdef int[::1] P_memview = P
    cdef int[::1] e_memview = e.reshape(-1)
    _mergePeriodNodes(d, M, &B_memview[0], &P_memview[0], &e_memview[0])
    return e

def countStiffMatData(B,P,e):
    M = e.shape[0]
    d = e.shape[1]-1
    N = P.size
    B = np.ascontiguousarray(B)
    P = np.ascontiguousarray(P)
    e = np.ascontiguousarray(e)
    cdef int[::1] B_memview = B
    cdef int[::1] P_memview = P
    cdef int[::1] e_memview = e.reshape(-1)
    return _countStiffMatData(d, M, N, 
            &B_memview[0], &P_memview[0], &e_memview[0])

def StiffMat(C_NUM,nu,B,P,e,E,eMeasure):
    M = e.shape[0]
    d = e.shape[1]-1
    N = P.size
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
    idx = _StiffMatOO(C_NUM, d, M, N, nu,
            &B_memview[0], &P_memview[0], &e_memview[0],
            &E_memview[0], &eMeasure_memview[0],
            &I_memview[0], &J_memview[0], &data_memview[0])
    print("idx=",idx,", C_NUM=",C_NUM)
    C = sp.sparse.coo_matrix((data,(I,J)),shape=[d*N+M,d*N+M])
    C = C.tocsr()
    return C

def countPoisson(B,P,e):
    M = e.shape[0]
    d = e.shape[1]-1
    N = P.size
    B = np.ascontiguousarray(B)
    P = np.ascontiguousarray(P)
    e = np.ascontiguousarray(e)
    cdef int[::1] B_memview = B
    cdef int[::1] P_memview = P
    cdef int[::1] e_memview = e.reshape(-1)
    return _countPoisson(d,M,N,
            &B_memview[0], &P_memview[0], &e_memview[0])

def Poisson(C_NUM,nu,B,P,e,E,eMeasure):
    M = e.shape[0]
    d = e.shape[1]-1
    N = P.size
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
    idx = _PoissonOO(C_NUM, d, M, N, nu,
            &B_memview[0], &P_memview[0], &e_memview[0],
            &E_memview[0], &eMeasure_memview[0],
            &I_memview[0], &J_memview[0], &data_memview[0])
    print("idx=",idx,", C_NUM=",C_NUM)
    C = sp.sparse.coo_matrix((data,(I,J)),shape=[N,N])
    C = C.tocsr()
    return C
