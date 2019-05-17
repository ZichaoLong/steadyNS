# distutils: language = c++

cdef extern from "steadyNS.h":
    cdef int _reduceP(const int N, long *P)
    cdef int _mergePeriodNodes(const int d, const int M,
            const long *B, const long *P, long *ep)

import numpy as np

def reduceP(P):
    if not P.flags['C_CONTIGUOUS']:
        P = np.ascontiguousarray(P)
    cdef long[::1] P_memview = P
    _reduceP(P.size,&P_memview[0])
    return P

def mergePeriodicNodes(B, P, e):
    M = e.shape[0]
    d = e.shape[1]-1
    if not B.flags['C_CONTIGUOUS']:
        B = np.ascontiguousarray(B)
    if not P.flags['C_CONTIGUOUS']:
        P = np.ascontiguousarray(P)
    if not e.flags['C_CONTIGUOUS']:
        e = np.ascontiguousarray(e)
    cdef long[::1] B_memview = B
    cdef long[::1] P_memview = P
    cdef long[::1] e_memview = e.reshape(-1)
    _mergePeriodNodes(d, M, &B_memview[0], &P_memview[0], &e_memview[0])
    return e

