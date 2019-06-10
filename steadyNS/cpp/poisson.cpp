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

int _Poisson_StiffMatOO_Boundary(const int d, const int N, const int NE, 
        const int *B, const int *P, 
        int &idx, int *I, int *J, double *data)
{
    // equations derived from boundary conditions
    for (int i=0; i<N+NE; ++i)
    {
        if (B[i]==4) // periodic nodes i; 
            // there are no periodic edges since there are no periodic nodes in e
            for (int l=0; l<d; ++l)
            {
                I[idx] = d*i+l;
                J[idx] = d*i+l;
                data[idx] = 1;
                ++idx;
                I[idx] = d*i+l;
                J[idx] = d*P[i]+l; // source nodes P[i]
                data[idx] = -1;
                ++idx;
            }
        else if (B[i]>0) // Dirichlet boundaries
            for (int l=0; l<d; ++l)
            {
                I[idx] = d*i+l;
                J[idx] = d*i+l;
                data[idx] = 1;
                ++idx;
            }
    }
    return 0;
}

int Poisson_CalculateTheta(const int d, const TensorAccessor<const double,2> &Ek,
        const int nQuad, TensorAccessor<const double,2> &Lambda,
        TensorAccessor<double,3> &Theta)
{
    for (int i=0; i<nQuad; ++i)
    {
        for (int j=0; j<d+1; ++j)
            for (int l=0; l<d; ++l)
                Theta[i][j][l] = 4*Lambda[i][j]*Ek[j][l]-Ek[j][l];
        for (int j1=1; j1<d+1; ++j1)
            for (int j2=0; j2<j1; ++j2)
            {
                for (int l=0; l<d; ++l)
                    Theta[i][d+1+j1*(j1-1)/2+j2][l] = 
                        4*(Lambda[i][j1]*Ek[j2][l]+Lambda[i][j2]*Ek[j1][l]);
            }
    }
    return 0;
}

int Poisson_UpdateStiffMatTheta2Sum(const int d, const int D, 
        const TensorAccessor<const double,2> &Ek, 
        const int nQuad2, const double *W2, TensorAccessor<const double,2> &Lambda2, 
        TensorAccessor<double,2> &Theta2Sum)
{
    Tensor<double,3> Theta2Tensor({nQuad2,D,d});
    TensorAccessor<double,3> Theta2 = Theta2Tensor.accessor();
    Poisson_CalculateTheta(d,Ek,nQuad2,Lambda2,Theta2);
    double tmp;
    // update Theta2Sum
    for (int j0=0; j0<D; ++j0)
        for (int j1=0; j1<D; ++j1)
        {
            tmp = 0;
            for (int i=0; i<nQuad2; ++i)
                for (int l=0; l<d; ++l)
                    tmp += W2[i]*Theta2[i][j0][l]*Theta2[i][j1][l];
            Theta2Sum[j0][j1] = tmp;
        }
    return 0;
}

int _Poisson_StiffMatOO_v(const int d, const int D, const int N, const int NE, 
        const int k, const double nu, const int *B, 
        const TensorAccessor<const int,1> &ek, const double ekMeasure, 
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
        }
    }
    return 0;
}

int _Poisson_StiffMatOO(const int C_NUM, const int d, const double nu,
        const int M, const int N, const int NE, 
        const int *B, const int *P, const int *ep, 
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
    // equations derived from boundary conditions
    _Poisson_StiffMatOO_Boundary(d, N, NE, B, P, idx, I, J, data);

    Tensor<double,2> Theta2SumTensor({D,D});
    TensorAccessor<double,2> Theta2Sum = Theta2SumTensor.accessor();
    // coefficients derived from test function in $e_k$
    for (int k=0; k<M; ++k) // $e_k$
    {
        // update Theta1Sum,Theta2Sum
        Poisson_UpdateStiffMatTheta2Sum(d, D, E[k], 
                nQuad2, W2, Lambda2, Theta2Sum);
        // stiffness matrix for $v^{j0,l}$, row=d*e[k][j0]+l
        _Poisson_StiffMatOO_v(d, D, N, NE, k, nu, B, 
                e[k], eMeasure[k], Theta2Sum, 
                idx, I, J, data);
    }
    return idx;
}

int _Poisson_countStiffMatData(const int d, const int M, const int N, const int NE,
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
        if (B[i]==4)
            COUNT += d*2;
        else if (B[i]>0)
            COUNT += d;
    }
#pragma omp parallel for schedule(static) reduction(+:COUNT)
    for (int k=0; k<M; ++k)
        for (int j0=0; j0<D; ++j0)
            if (B[e[k][j0]]>0) 
                continue;
            else 
                COUNT += d*D;
    return COUNT;
}
