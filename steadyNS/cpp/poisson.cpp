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


int _countPoisson(const int d, const int M, const int N,
        const int *B, const int *P, const int *ep)
{
    int COUNT=0;
    int esize[] = {M,d+1};
    int estride[] = {d+1,1};
    TensorAccessor<const int,2> e(ep,esize,estride);
#pragma omp parallel for schedule(static) reduction(+:COUNT)
    for (int i=0; i<N; ++i)
    {
        if (B[i]==0 || B[i]==-1) continue;
        else if (B[i]==4) COUNT +=2;
        else ++COUNT;
    }
#pragma omp parallel for schedule(static) reduction(+:COUNT)
    for (int k=0; k<M; ++k) // $e_k$
        for (int j=0,i; j<d+1; ++j)
        {
            i = e[k][j];
            if (B[i]==1 || B[i]==2 || B[i]==3)
                continue; // no test function $v_l^j$ here
            COUNT += d+1;
        }
    return COUNT;
}

int _PoissonOO(const int C_NUM, const int d, const int M, const int N, const double nu, 
        const int *B, const int *P, const int *ep, const double *Ep, const double *eMeasure, 
        int *I, int *J, double *data)
{
    int esize[] = {M,d+1};
    int estride[] = {d+1,1};
    TensorAccessor<const int,2> e(ep,esize,estride);
    int Esize[] = {M,d+1,d};
    int Estride[] = {(d+1)*d,d,1};
    TensorAccessor<const double,3> E(Ep,Esize,Estride);
    double eMeasureMean = 0;
    for (int k=0; k<M; ++k)
        eMeasureMean += eMeasure[k];
    eMeasureMean /= M;
    int idx = 0;
    for (int i=0; i<N; ++i)
    {
        if (B[i]==0 || B[i]==-1)
            continue;
        else if (B[i]==4)
        {
            I[idx] = i;
            J[idx] = i;
            data[idx] = eMeasureMean;
            ++idx;
            I[idx] = i;
            J[idx] = P[i];
            data[idx] = -eMeasureMean;
            ++idx;
        }
        else // (B[i]==1 || B[i]==2 || B[i]==3) // no degrees of freedom
        {
            I[idx] = i;
            J[idx] = i;
            data[idx] = eMeasureMean;
            ++idx;
        }
    }
    Tensor<double,2> EETTensor({d+1,d+1});
    TensorAccessor<double,2> EET = EETTensor.accessor();
    for (int k=0; k<M; ++k) // $e_k$
    {
        // EET = matmul(E[k],transpose(E[k])); // (d+1)x(d+1)
        for (int j1=0; j1<d+1; ++j1)
            for (int j2=0; j2<d+1; ++j2)
            {
                EET[j1][j2] = 0;
                for (int l=0; l<d; ++l)
                    EET[j1][j2] += E[k][j1][l]*E[k][j2][l];
            }
        for (int j=0,i; j<d+1; ++j)
        {
            i = e[k][j];
            if (B[i]==1 || B[i]==2 || B[i]==3)
                continue; // no test function $v_l^j$ here
            if (B[i]==4)
                cout << "error" << endl;
            for (int je=0; je<d+1; ++je)
            {
                I[idx] = i;
                J[idx] = e[k][je];
                data[idx] = nu*eMeasure[k]*EET[je][j];
                ++idx;
            }
        }
    }
    return idx;
}
