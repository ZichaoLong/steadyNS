# distutils: language = c++
# cython: language_level=3

cdef extern from "steadyNS.h":
    int _UGU(const int d, const int M, const int N, const int NE, 
            const int *ep, const double *Ep, const double *eMeasure, 
            const int nQuad5, const double *W5, const double *Lambda5p, 
            const double *Up, double *ugu)
    int _sourceFOO(const int C_NUM, const int d, const int M, const int N, const int NE, 
            const int *ep, const double *Ep, const double *eMeasure, 
            const int nQuad4, const double *W4, const double *Lambda4p, 
            int *I, int *J, double *data)
    int _QGUOO(const int C_NUM, const int d, const int M, const int N, const int NE,
            const int *ep, const double *Ep, const double *eMeasure,
            const int nQuad2, const double *W2, const double *Lambda2p,
            int *I, int *J, double *data)

