/*************************************************************************
  > File Name: steadyNSCPP.h
  > Author: Long Zichao
  > Mail: zlong@pku.edu.cn
  > Created Time: 2019-05-16
 ************************************************************************/

#ifndef STEADYNS_H
#define STEADYNS_H

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

#ifdef __cplusplus
}
#endif
#endif // STEADYNS_H
