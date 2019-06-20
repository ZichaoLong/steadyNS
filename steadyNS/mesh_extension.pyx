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

def switchEdgeNode(Edge):
    L = Edge.shape[0]
    Edge = np.ascontiguousarray(Edge)
    cdef int[::1] Edge_memview = Edge.reshape(-1)
    _switchEdgeNode(L, &Edge_memview[0])
    return Edge

def updateEdgeTags(d,Edge,B,coord):
    N = B.shape[0]
    NE = Edge.shape[0]
    Edge = np.ascontiguousarray(Edge)
    B = np.ascontiguousarray(B)
    coord = np.ascontiguousarray(coord)
    Bedge = np.zeros(NE,dtype=np.int32)
    cdef int[::1] Edge_memview = Edge.reshape(-1)
    cdef int[::1] B_memview = B
    cdef int[::1] Bedge_memview = Bedge
    cdef double[::1] coord_memview = coord.reshape(-1)
    _updateEdgeTags(d, N, NE, &Edge_memview[0], &B_memview[0], &coord_memview[0], &Bedge_memview[0])
    return Bedge
