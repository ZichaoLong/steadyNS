/*************************************************************************
  > File Name: steadyNSCPP.cpp
  > Author: Long Zichao
  > Mail: zlong@pku.edu.cn
  > Created Time: 2019-05-16
 ************************************************************************/

#include<iostream>
#include<vector>
#include<cmath>
#include "steadyNS.h"
#include "ASTen/TensorAccessor.h"
using std::cout; using std::endl; using std::ends;

int _reduceP(const int N, long *P)
{
    for (int i=0; i<N; ++i)
    {
        if (P[i]!=-1)
            if (P[P[i]]!=-1) P[i] = P[P[i]];
    }
    return 0;
}

int _mergePeriodNodes(const int d, const int M,
        const long *B, const long *P, long *ep)
{
    int esize[] = {M,d+1};
    int estride[] = {d+1,1};
    TensorAccessor<long,2> e(ep,esize,estride);
#pragma omp parallel for schedule(static)
    for (int k=0; k<M; ++k)
        for (int j=0; j<d+1; ++j)
            if (B[e[k][j]]==4)
                e[k][j] = P[e[k][j]];
    return 0;
}
