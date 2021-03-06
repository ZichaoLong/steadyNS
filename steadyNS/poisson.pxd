# distutils: language = c++
# cython: language_level=3

cdef extern from "steadyNS.h":
    int _P2StiffMatOO(const int C_NUM, const int d,
            const int M, const int N, const int NE, 
            const int *ep, const double *Ep, const double *eMeasure, 
            const int nQuad2, const double *W2, const double *Lambda2p, 
            int *I, int *J, double *data)
