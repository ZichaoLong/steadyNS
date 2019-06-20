/*************************************************************************
  > File Name: GlobalStiffnessMatrix.cpp
  > Author: Long Zichao
  > Mail: zlong@pku.edu.cn
  > Created Time: 2019-05-18
 ************************************************************************/

#include<iostream>
#include<cmath>
#include "steadyNS.h"
#include "ASTen/TensorAccessor.h"
#include "ASTen/Tensor.h"
using std::cout; using std::endl; using std::ends;

int _StiffMatOO_v(const int d, const int D, const int N, const int NE, 
        const int k, const double nu, const int *B, 
        const TensorAccessor<const int,1> &ek, const double ekMeasure, 
        TensorAccessor<double,2> &Theta1Sum, 
        TensorAccessor<double,2> &Theta2Sum, 
        int &idx, int *I, int *J, double *data)
{
    int row=-1;
    // stiffness matrix for $v^{j0,l}$, row=d*ek[j0]+l
    for (int j0=0; j0<D; ++j0)
    {
        if (B[ek[j0]]>0) continue; // boundary equations have been set done
        for (int l=0; l<d; ++l)
        {
            row = d*ek[j0]+l;
            for (int j1=0; j1<D; ++j1)
            {
                I[idx] = row;
                J[idx] = d*ek[j1]+l;
                data[idx] = nu*ekMeasure*Theta2Sum[j0][j1];
                ++idx;
            }
            I[idx] = row;
            J[idx] = d*(N+NE)+k;
            data[idx] = -ekMeasure*Theta1Sum[j0][l];
            ++idx;
        }
    }
    return 0;
}

int _StiffMatOO_q(const int d, const int D, const int M, const int N, const int NE, const int k, 
        const TensorAccessor<const int,1> &ek, TensorAccessor<double,2> &Theta1Sum, 
        int &idx, int *I, int *J, double *data)
{
    int row;
    // stiffness matrix for $q_k-q_{k-1}$, row=d*(N+NE)+k-1, 
    if (k>0)
    {
        row = d*(N+NE)+k-1;
        for (int j=0; j<D; ++j)
            for (int l=0; l<d; ++l)
            {
                I[idx] = row;
                J[idx] = d*ek[j]+l;
                data[idx] = Theta1Sum[j][l];
                ++idx;
            }
    }
    // stiffness matrix for $q_{k+1}-q_k$, row=d*(N+NE)+k, 
    if (k<M-1)
    {
        row = d*(N+NE)+k;
        for (int j=0; j<D; ++j)
            for (int l=0; l<d; ++l)
            {
                I[idx] = row;
                J[idx] = d*ek[j]+l;
                data[idx] = -Theta1Sum[j][l];
                ++idx;
            }
    }
    return 0;
}

int _StiffMatOO(const int C_NUM, const int d, const double nu, 
        const int M, const int N, const int NE, 
        const int *B, const int *P, const int *ep, 
        const double *Ep, const double *eMeasure, 
        const int nQuad1, const double *W1, const double *Lambda1p, 
        const int nQuad2, const double *W2, const double *Lambda2p, 
        int *I, int *J, double *data)
{
    int D = (d+1)*(d+2)/2;
    // convert pointer to TensorAccessor
    int esize[] = {M,D}; int estride[] = {D,1};
    TensorAccessor<const int,2> e(ep,esize,estride);
    int Esize[] = {M,d+1,d}; int Estride[] = {(d+1)*d,d,1};
    TensorAccessor<const double,3> E(Ep,Esize,Estride);
    int Lambda1size[] = {nQuad1,d+1}; int Lambda1stride[] = {d+1,1};
    TensorAccessor<const double,2> Lambda1(Lambda1p,Lambda1size,Lambda1stride);
    int Lambda2size[] = {nQuad2,d+1}; int Lambda2stride[] = {d+1,1};
    TensorAccessor<const double,2> Lambda2(Lambda2p,Lambda2size,Lambda2stride);

    int idx = 0;

    // equations derived from boundary conditions
    _StiffMatOO_Boundary(d, N, NE, B, P, idx, I, J, data);

    Tensor<double,2> Theta1SumTensor({D,d});
    Tensor<double,2> Theta2SumTensor({D,D});
    TensorAccessor<double,2> Theta1Sum = Theta1SumTensor.accessor();
    TensorAccessor<double,2> Theta2Sum = Theta2SumTensor.accessor();
    // coefficients derived from test function in $e_k$
    for (int k=0; k<M; ++k) // $e_k$
    {
        // update Theta1Sum,Theta2Sum
        UpdateStiffMatTheta1Sum(d, D, E[k], nQuad1, W1, Lambda1, Theta1Sum);
        UpdateStiffMatTheta2Sum(d, D, E[k], nQuad2, W2, Lambda2, Theta2Sum);
        // stiffness matrix for $v^{j0,l}$, row=d*e[k][j0]+l
        _StiffMatOO_v(d, D, N, NE, k, nu, B, 
                e[k], eMeasure[k], Theta1Sum, Theta2Sum, 
                idx, I, J, data);
        // stiffness matrix for $q_k-q_{k-1}$ and $q_{k+1}-q_k$, 
        // row=d*(N+NE)+k-1 and d*(N+NE)+k
        _StiffMatOO_q(d, D, M, N, NE, k, e[k], Theta1Sum, 
                idx, I, J, data);
    }
    I[idx] = d*(N+NE)+M-1;
    J[idx] = d*(N+NE)+M-1;
    data[idx] = 1;
    ++idx;
    return idx;
}

int _countStiffMatData(const int d, const int M, const int N, const int NE,
        const int *B, const int *P, const int *ep)
{
    int COUNT=0;
    int D = (d+1)*(d+2)/2;
    // convert pointer to TensorAccessor
    int esize[] = {M,D}; int estride[] = {D,1};
    TensorAccessor<const int,2> e(ep,esize,estride);
#pragma omp parallel for schedule(static) reduction(+:COUNT)
    for (int i=0; i<N+NE; ++i)
    {
        if (B[i]>0)
            COUNT += d;
    }
#pragma omp parallel for schedule(static) reduction(+:COUNT)
    for (int k=0; k<M; ++k)
        for (int j0=0; j0<D; ++j0)
        {
            if (B[e[k][j0]]>0) continue;
            COUNT += d*(D+1);
        }
    COUNT += 2*(M-1)*D*d+1;
    return COUNT;
}
