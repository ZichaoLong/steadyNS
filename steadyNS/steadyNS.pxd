# distutils: language = c++
# cython: language_level=3

cdef extern from "steadyNS.h":
    int _countStiffMatData(const int d, const int M, const int N, const int NE,
            const int *B, const int *ep)
    int _StiffMatOO(const int C_NUM, const int d, const double nu, 
            const int M, const int N, const int NE, 
            const int *B, const int *ep, 
            const double *Ep, const double *eMeasure, 
            const int nQuad1, const double *W1, const double *Lambda1p, 
            const int nQuad2, const double *W2, const double *Lambda2p, 
            int *I, int *J, double *data)
    int _RHI(const int d, const int M, const int N, const int NE, 
            const int *B, const int *ep, const double *Ep, const double *eMeasure, 
            const int nQuad5, const double *W5, const double *Lambda5p, 
            const double *UPp, double *rhi)

