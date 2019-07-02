# distutils: language = c++
# cython: language_level=3

cdef extern from "steadyNS.h":
    int _StiffMatOO(const int d, const int M, const int N, const int NE, 
            const int *B, const int *ep, const double *Ep, const double *eMeasure, 
            const int nQuad1, const double *W1, const double *Lambda1p, 
            int *I, int *J, double *data)
    int _countStiffMatUplusGU(const int d, const int M, const int N, const int NE, 
            const int *B, const int *ep)
    int _countStiffMatUGUplus(const int d, const int M, const int N, const int NE, 
            const int *B, const int *ep)
    int _UGU(const int C_NUM_UplusGU, const int C_NUM_UGUplus, 
            const int d, const int M, const int N, const int NE, 
            const int *B, const int *ep, const double *Ep, const double *eMeasure, 
            const int nQuad5, const double *W5, const double *Lambda5p, 
            const double *Up, 
            double *ugu, 
            int *IUplusGU, int *JUplusGU, double *dataUplusGU, 
            int *IUGUplus, int *JUGUplus, double *dataUGUplus)
    int _sourceF(const int d, const int M, const int N, const int NE, 
            const int *B, const int *ep, const double *Ep, const double *eMeasure, 
            const int nQuad4, const double *W4, const double *Lambda4p, 
            const double *Fp, double *rhi)

