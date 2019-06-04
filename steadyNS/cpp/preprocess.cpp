/*************************************************************************
  > File Name: preprocesscpp.cpp
  > Author: Long Zichao
  > Mail: zlong@pku.edu.cn
  > Created Time: 2019-05-16
 ************************************************************************/

#include<iostream>
#include<cmath>
#include "steadyNS.h"
#include "ASTen/TensorAccessor.h"
using std::cout; using std::endl; using std::ends;

int _reduceP(const int N, int *P)
{
    for (int i=0; i<N; ++i)
    {
        if (P[i]!=-1)
            if (P[P[i]]!=-1) P[i] = P[P[i]];
    }
    return 0;
}

int _mergePeriodNodes(const int d, const int M,
        const int *B, const int *P, int *ep)
{
    int esize[] = {M,d+1};
    int estride[] = {d+1,1};
    TensorAccessor<int,2> e(ep,esize,estride);
#pragma omp parallel for schedule(static)
    for (int k=0; k<M; ++k)
        for (int j=0; j<d+1; ++j)
            if (B[e[k][j]]==4)
                e[k][j] = P[e[k][j]];
    return 0;
}

int _countStiffMatData(const int d, const int M, const int N,
        const int *B, const int *P, const int *ep)
{
    int COUNT=0;
    int esize[] = {M,d+1};
    int estride[] = {d+1,1};
    TensorAccessor<const int,2> e(ep,esize,estride);
#pragma omp parallel for schedule(static) reduction(+:COUNT)
    for (int i=0; i<N; ++i)
    {
        if (B[i]==4)
            COUNT += d*2;
        if (B[i]==1 || B[i]==2 || B[i]==3)
            COUNT += d;
    }
#pragma omp parallel for schedule(static) reduction(+:COUNT)
    for (int k=0; k<M; ++k) 
        for (int j=0,i; j<d+1; ++j)
        {
            i = e[k][j];
            if (B[i]==1 || B[i]==2 || B[i]==3)
                continue; 
            COUNT += d*(d+2);
        }
    COUNT += 2*(M-1)*(d+1)*d+M;
    return COUNT;
}
