# distutils: language = c++
# cython: language_level=3

cdef extern from "steadyNS.h":
    int _reduceP(const int N, int *P)
    int _mergePeriodNodes(const int d, const int M,
       const int *B, const int *P, int *ep)

