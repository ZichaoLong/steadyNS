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

int _StiffMatOO(const int d, const int M, const int N, const int NE, 
        const int *B, const int *ep, const double *Ep, const double *eMeasure, 
        const int nQuad1, const double *W1, const double *Lambda1p, 
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

    int idx = 0;

    Tensor<double,2> Theta1SumTensor({D,d});
    TensorAccessor<double,2> Theta1Sum = Theta1SumTensor.accessor();
    // coefficients derived from test function in $e_k$
    for (int k=0; k<M; ++k) // $e_k$
    {
        // update Theta1Sum
        UpdateStiffMatTheta1Sum(d, D, E[k], nQuad1, W1, Lambda1, Theta1Sum);
        // stiffness matrix for $q_k$, 
        const int *ek = e[k].data();
        for (int j=0; j<D; ++j)
            for (int l=0; l<d; ++l)
            {
                I[idx] = k;
                J[idx] = ek[j];
                data[idx] = eMeasure[k]*Theta1Sum[j][l];
                ++idx;
            }
    }
    return idx;
}

