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


int _reduceP(const int N, int *P);
int _mergePeriodNodes(const int d, const int M,
        const int *B, const int *P, int *ep);
int _switchEdgeNode(const int L, int *Edge);
int _updateEdgeTags(const int N, const int NE, const int *Edge, 
        const int *B, int *Bedge);
int _countStiffMatData(const int d, const int M, const int N, const int NE,
        const int *B, const int *P, const int *ep);
int _StiffMatOO(const int C_NUM, const int d, const double nu, 
        const int M, const int N, const int NE, 
        const int *B, const int *P, const int *ep, 
        const double *Ep, const double *eMeasure, 
        const int nQuad1, const double *W1, const double *Lambda1p, 
        const int nQuad2, const double *W2, const double *Lambda2p, 
        int *I, int *J, double *data);
int _Poisson_countStiffMatData(const int d, const int M, const int N, const int NE,
        const int *B, const int *P, const int *ep);
int _Poisson_StiffMatOO(const int C_NUM, const int d, const double nu,
        const int M, const int N, const int NE, 
        const int *B, const int *P, const int *ep, 
        const double *Ep, const double *eMeasure, 
        const int nQuad2, const double *W2, const double *Lambda2p, 
        int *I, int *J, double *data);

int CalculateTheta(const int d, const TensorAccessor<const double,2> &Ek,
        const int nQuad, TensorAccessor<const double,2> &Lambda,
        TensorAccessor<double,3> &Theta);
int _StiffMatOO_Boundary(const int d, const int N, const int NE, 
        const int *B, const int *P, 
        int &idx, int *I, int *J, double *data);
int UpdateStiffMatTheta1Sum(const int d, const int D, const TensorAccessor<const double,2> &Ek,
        const int nQuad1, const double *W1, TensorAccessor<const double,2> &Lambda1,
        TensorAccessor<double,2> &Theta1Sum);
int UpdateStiffMatTheta2Sum(const int d, const int D, const TensorAccessor<const double,2> &Ek, 
        const int nQuad2, const double *W2, TensorAccessor<const double,2> &Lambda2, 
        TensorAccessor<double,2> &Theta2Sum);

#ifdef __cplusplus
}
#endif
#endif // STEADYNS_H
