/*************************************************************************
  > File Name: poisson.cpp
  > Author: Long Zichao
  > Mail: zlong@pku.edu.cn
  > Created Time: 2019-05-19
 ************************************************************************/

#include<iostream>
#include<cmath>
#include "steadyNS.h"
#include "ASTen/TensorAccessor.h"
#include "ASTen/Tensor.h"
using std::cout; using std::endl; using std::ends;

int _Poisson_StiffMatOO_v(const int d, const int D, const int N, const int NE, 
        const int k, const double nu, const int *B, 
        const TensorAccessor<const int,1> &ek, const double ekMeasure, 
        TensorAccessor<double,2> &Theta2Sum, 
        int &idx, int *I, int *J, double *data)
{
    int row=-1;
    // stiffness matrix for $v^{j0}$, row=ek[j0]
    for (int j0=0; j0<D; ++j0)
    {
        if (B[ek[j0]]>0) continue; 
        row = ek[j0];
        for (int j1=0; j1<D; ++j1)
        {
            I[idx] = row;
            J[idx] = ek[j1];
            data[idx] = nu*ekMeasure*Theta2Sum[j0][j1];
            ++idx;
        }
    }
    return 0;
}

int _Poisson_StiffMatOO(const int C_NUM, const int d, const double nu,
        const int M, const int N, const int NE, 
        const int *B, const int *ep, 
        const double *Ep, const double *eMeasure, 
        const int nQuad2, const double *W2, const double *Lambda2p, 
        int *I, int *J, double *data)
{
    int D = (d+1)*(d+2)/2;
    // convert pointer to TensorAccessor
    int esize[] = {M,D}; int estride[] = {D,1};
    TensorAccessor<const int,2> e(ep,esize,estride);
    int Esize[] = {M,d+1,d}; int Estride[] = {(d+1)*d,d,1};
    TensorAccessor<const double,3> E(Ep,Esize,Estride);
    int Lambda2size[] = {nQuad2,d+1}; int Lambda2stride[] = {d+1,1};
    TensorAccessor<const double,2> Lambda2(Lambda2p,Lambda2size,Lambda2stride);

    int idx = 0;

    Tensor<double,2> Theta2SumTensor({D,D});
    TensorAccessor<double,2> Theta2Sum = Theta2SumTensor.accessor();
    // coefficients derived from test function in $e_k$
    for (int k=0; k<M; ++k) // $e_k$
    {
        // update Theta2Sum
        UpdateStiffMatTheta2Sum(d, D, E[k], nQuad2, W2, Lambda2, Theta2Sum);
        // stiffness matrix for $v^{j0}$, row=e[k][j0]
        _Poisson_StiffMatOO_v(d, D, N, NE, k, nu, B, 
                e[k], eMeasure[k], Theta2Sum, 
                idx, I, J, data);
    }
    return idx;
}

int _Poisson_countStiffMatData(const int d, const int M, const int N, const int NE,
        const int *B, const int *ep)
{
    int COUNT=0;
    int D = (d+1)*(d+2)/2;
    // convert pointer to TensorAccessor
    int esize[] = {M,D}; int estride[] = {D,1};
    TensorAccessor<const int,2> e(ep,esize,estride);
#pragma omp parallel for schedule(static) reduction(+:COUNT)
    for (int k=0; k<M; ++k)
        for (int j0=0; j0<D; ++j0)
            if (B[e[k][j0]]>0) 
                continue;
            else 
                COUNT += D;
    return COUNT;
}
