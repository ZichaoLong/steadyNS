/*************************************************************************
  > File Name: steadyNSCPP.h
  > Author: Long Zichao
  > Mail: zlong@pku.edu.cn
  > Created Time: 2019-05-16
 ************************************************************************/

#ifndef STEADYNSCPP_H
#define STEADYNSCPP_H

int _reduceP(const int N, int *P);
int _mergePeriodNodes(const int d, const int M,
        const int *B, const int *P, int *ep);
int _countStiffMatData(const int d, const int M, const int N,
        const int *B, const int *P, const int *ep);
int _StiffMatOO(const int C_NUM, const int d, const int M, const int N, const double nu, 
        const int *B, const int *P, const int *ep, const double *Ep, const double *eMeasure, 
        int *I, int *J, double *data);
int _countPoisson(const int d, const int M, const int N,
        const int *B, const int *P, const int *ep);
int _PoissonOO(const int C_NUM, const int d, const int M, const int N, const double nu, 
        const int *B, const int *P, const int *ep, const double *Ep, const double *eMeasure, 
        int *I, int *J, double *data);

#endif // STEADYNSCPP_H
