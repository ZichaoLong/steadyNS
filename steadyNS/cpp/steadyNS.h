/*************************************************************************
  > File Name: steadyNSCPP.h
  > Author: Long Zichao
  > Mail: zlong@pku.edu.cn
  > Created Time: 2019-05-16
 ************************************************************************/

#ifndef STEADYNS_H
#define STEADYNS_H

#include "ASTen/TensorAccessor.h"
#include "ASTen/Tensor.h"

#ifdef __cplusplus
extern "C" {
#endif


int _switchEdgeNode(const int L, int *Edge);
int _updateEdgeTags(const int d, const int N, const int NE, const int *Edge, 
        const int *B, const double *coord, int *Bedge);

int _P2StiffMatOO(const int C_NUM, const int d,
        const int M, const int N, const int NE, 
        const int *ep, const double *Ep, const double *eMeasure, 
        const int nQuad2, const double *W2, const double *Lambda2p, 
        int *I, int *J, double *data);

int _QGUOO(const int C_NUM, const int d, const int M, const int N, const int NE,
        const int *ep, const double *Ep, const double *eMeasure,
        const int nQuad2, const double *W2, const double *Lambda2p,
        int *I, int *J, double *data);
int _UGU(const int d, const int M, const int N, const int NE, 
        const int *ep, const double *Ep, const double *eMeasure, 
        const int nQuad5, const double *W5, const double *Lambda5p, 
        const double *Up, double *ugu);
int _sourceFOO(const int C_NUM, const int d, const int M, const int N, const int NE, 
        const int *ep, const double *Ep, const double *eMeasure, 
        const int nQuad4, const double *W4, const double *Lambda4p, 
        int *I, int *J, double *data);

int _CsrMulVec(const int M, const int N, const int nnz, 
        const int *IA, const int *JA, const double *data, 
        const double *x, double *y);
int UpdateStiffMatTheta1Sum(const int d, const int D, const TensorAccessor<const double,2> &Ek,
        const int nQuad1, const double *W1, const TensorAccessor<const double,2> &Lambda1,
        TensorAccessor<double,2> &Theta1Sum);
int UpdateStiffMatTheta2Sum(const int d, const int D, const TensorAccessor<const double,2> &Ek, 
        const int nQuad2, const double *W2, const TensorAccessor<const double,2> &Lambda2, 
        TensorAccessor<double,2> &Theta2Sum);
int CalculateTheta(const int d, const TensorAccessor<const double,2> &Ek,
        const int nQuad, const TensorAccessor<const double,2> &Lambda,
        TensorAccessor<double,3> &Theta);
int CalculateGamma(const int d, const int D, const int nQuad, 
        const TensorAccessor<const double,2> &Lambda, 
        TensorAccessor<double,2> Gamma);
int CalculateU(const int d, const int D, const int nQuad, 
        const TensorAccessor<const double,2> &Gamma, 
        const TensorAccessor<const double,2> &Ue, 
        TensorAccessor<double,2> &U);
int CalculateGU(const int d, const int D, const int nQuad, 
        const TensorAccessor<const double,3> &Theta, 
        const TensorAccessor<const double,2> &Ue, 
        TensorAccessor<double,3> &GU);
int CalculateUGU(const int d, const int nQuad, 
        const TensorAccessor<const double,2> &U, 
        const TensorAccessor<const double,3> &GU, 
        TensorAccessor<double,2> &UGU);
int CalculateTrGU(const int d, const int nQuad, 
        const TensorAccessor<const double,3> &GU, 
        TensorAccessor<double,1> &TrGU);

int _InterpP2ToUniformGrid(const double dx,
        const int xidxrange, const int yidxrange, const int zidxrange,
        const int M, const int N, const int NE,
        const int *ep , const double *basisp, const double *coordAllp,
        const double *Up, double *uniformUp);
int _InterpP1ToUniformGrid(const double dx,
        const int xidxrange, const int yidxrange, const int zidxrange,
        const int M, const int N, const int NE,
        const int *ep , const double *basisp, const double *coordAllp,
        const double *Pp, double *uniformPp);

#ifdef __cplusplus
}
#endif
#endif // STEADYNS_H
