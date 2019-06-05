# distutils: language = c++
# cython: language_level=3

import numpy as np

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

