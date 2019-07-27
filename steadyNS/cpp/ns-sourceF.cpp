/*************************************************************************
  > File Name: ns-sourceF.cpp
  > Author: Long Zichao
  > Mail: zlong@pku.edu.cn
  > Created Time: 2019-07-02
 ************************************************************************/

#include<iostream>
#include<cmath>
#include "steadyNS.h"
#include "ASTen/TensorAccessor.h"
#include "ASTen/Tensor.h"
using std::cout; using std::endl; using std::ends;

int _sourceFOO(const int C_NUM, const int d, const int M, const int N, const int NE, 
        const int *ep, const double *Ep, const double *eMeasure, 
        const int nQuad4, const double *W4, const double *Lambda4p, 
        int *I, int *J, double *data)
{
    int D = (d+1)*(d+2)/2;
    // convert pointer to TensorAccessor
    int esize[] = {M,D}; int estride[] = {D,1};
    TensorAccessor<const int,2> e(ep,esize,estride);
    int Esize[] = {M,d+1,d}; int Estride[] = {(d+1)*d,d,1};
    TensorAccessor<const double,3> E(Ep,Esize,Estride);
    int Lambda4size[] = {nQuad4,d+1}; int Lambda4stride[] = {d+1,1};
    TensorAccessor<const double,2> Lambda4(Lambda4p,Lambda4size,Lambda4stride);
    
    Tensor<double,2> Gamma4Tensor({nQuad4,D});
    TensorAccessor<double,2> Gamma4 = Gamma4Tensor.accessor();
    CalculateGamma(d, D, nQuad4, Lambda4.ConstAccessor(), Gamma4);

    Tensor<double,2> VU4Tensor({D,D});
    TensorAccessor<double,2> VU4 = VU4Tensor.accessor();
    for (int j0=0; j0<D; ++j0)
        for (int j1=0; j1<D; ++j1)
        {
            VU4[j0][j1] = 0;
            for (int i=0; i<nQuad4; ++i)
                VU4[j0][j1] += W4[i]*Gamma4[i][j0]*Gamma4[i][j1];
        }

    int idx = 0;
    for (int k=0; k<M; ++k)
    {
        const int *ek = e[k].data();
        for (int j0=0; j0<D; ++j0)
        {
            int row = ek[j0];
            for (int j1=0; j1<D; ++j1)
            {
                I[idx] = row;
                J[idx] = ek[j1];
                data[idx] = eMeasure[k]*VU4[j0][j1];
                ++idx;
            }
        }
    }
    return idx;
}

