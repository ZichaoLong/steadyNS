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

int _QGUOO(const int C_NUM, const int d, const int M, const int N, const int NE,
        const int *ep, const double *Ep, const double *eMeasure,
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

    // coefficients derived from test function in $e_k$
    for (int k=0; k<M; ++k) // $e_k$
    {
        Tensor<double,3> Theta2Tensor({nQuad2,D,d});
        TensorAccessor<double,3> Theta2 = Theta2Tensor.accessor();
        CalculateTheta(d, E[k], nQuad2, Lambda2.ConstAccessor(), Theta2);
        Tensor<double,3> Theta2SumQGUTensor({d+1,D,d});
        TensorAccessor<double,3> Theta2SumQGU = Theta2SumQGUTensor.accessor();
        for (int j0=0; j0<d+1; ++j0)
            for (int j1=0; j1<D; ++j1)
                for (int l=0; l<d; ++l)
                {
                    Theta2SumQGU[j0][j1][l] = 0;
                    for (int i=0; i<nQuad2; ++i)
                        Theta2SumQGU[j0][j1][l] += W2[i]*Lambda2[i][j0]*Theta2[i][j1][l];
                }
        // stiffness matrix for $q_k$, 
        const int *ek = e[k].data();
        for (int j0=0; j0<d+1; ++j0)
        {
            int row = ek[j0];
            for (int j1=0; j1<D; ++j1)
            {
                int col=ek[j1];
                for (int l=0; l<d; ++l)
                {
                    I[idx] = row;
                    J[idx] = col;
                    data[idx] = eMeasure[k]*Theta2SumQGU[j0][j1][l];
                    ++idx;
                }
            }
        }
    }
    return idx;
}

