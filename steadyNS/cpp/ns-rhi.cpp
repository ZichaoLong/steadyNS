/*************************************************************************
  > File Name: ns-rhi.cpp
  > Author: Long Zichao
  > Mail: zlong@pku.edu.cn
  > Created Time: 2019-06-12
 ************************************************************************/

#include<iostream>
#include<cmath>
#include "steadyNS.h"
#include "ASTen/TensorAccessor.h"
#include "ASTen/Tensor.h"
using std::cout; using std::endl; using std::ends;

int _RHI_v(const int d, const int D, const int *B, 
        const TensorAccessor<const int,1> &ek, const double ekMeasure, 
        TensorAccessor<double,4> VUTheta5, 
        const double *UPp, double *rhi)
{
    int row=-1;
    double tmp = 0;
    // stiffness matrix for $v^{j0,l}$, row=d*ek[j0]+l
    for (int j0=0; j0<D; ++j0)
    {
        if (B[ek[j0]]>0) continue; // boundary equations have been set done
        for (int l=0; l<d; ++l)
        {
            tmp = 0;
            for (int j1=0; j1<D; ++j1)
                for (int j2=0; j2<D; ++j2)
                    for (int l1=0; l1<d; ++l1)
                        tmp += VUTheta5[j0][j1][j2][l1]
                            *(2*UPp[d*ek[j1]+l1]*UPp[d*ek[j2]+l]
                             +UPp[d*ek[j2]+l1]*UPp[d*ek[j1]+l]);
            row = d*ek[j0]+l;
            rhi[row] -= 0.5*ekMeasure*tmp;
        }
    }
    return 0;
}

int _RHI(const int d, const int M, const int N, const int NE, 
        const int *B, const int *ep, const double *Ep, const double *eMeasure, 
        const int nQuad5, const double *W5, const double *Lambda5p, 
        const double *UPp, double *rhi)
{
    int D = (d+1)*(d+2)/2;
    // convert pointer to TensorAccessor
    int esize[] = {M,D}; int estride[] = {D,1};
    TensorAccessor<const int,2> e(ep,esize,estride);
    int Esize[] = {M,d+1,d}; int Estride[] = {(d+1)*d,d,1};
    TensorAccessor<const double,3> E(Ep,Esize,Estride);
    int Lambda5size[] = {nQuad5,d+1}; int Lambda5stride[] = {d+1,1};
    TensorAccessor<const double,2> Lambda5(Lambda5p,Lambda5size,Lambda5stride);

#pragma omp parallel for schedule(static)
    for (int i=0; i<d*(N+NE)+M; ++i)
        rhi[i] = 0; // initialize all elements as 0

    _RHI_Boundary_v(d, M, N, NE, B, rhi);

    Tensor<double,3> VU5Tensor({nQuad5,D,D});
    TensorAccessor<double,3> VU5 = VU5Tensor.accessor();
    CalculateVU(d, D, nQuad5, W5, Lambda5, VU5);

    // coefficients derived from test function in $e_k$
    for (int k=0; k<M; ++k)
    {
        Tensor<double,3> Theta5Tensor({nQuad5,D,d});
        TensorAccessor<double,3> Theta5 = Theta5Tensor.accessor();
        CalculateTheta(d, E[k], nQuad5, Lambda5, Theta5);
        Tensor<double,4> VUTheta5Tensor({D,D,D,d});
        TensorAccessor<double,4> VUTheta5 = VUTheta5Tensor.accessor();
        double tmp = 0;
        for (int j0=0; j0<D; ++j0)
            for (int j1=0; j1<D; ++j1)
                for (int j2=0; j2<D; ++j2)
                    for (int l=0; l<d; ++l)
                    {
                        tmp = 0;
                        for (int i=0; i<nQuad5; ++i)
                            tmp += VU5[i][j0][j1]*Theta5[i][j2][l];
                        VUTheta5[j0][j1][j2][l] = tmp;
                    }
        _RHI_v(d, D, B, e[k], eMeasure[k], VUTheta5, UPp, rhi);
    }
    return 0;
}
