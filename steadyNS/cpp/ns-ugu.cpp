/*************************************************************************
  > File Name: ns-ugu.cpp
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

int _UGU(const int d, const int M, const int N, const int NE, 
        const int *ep, const double *Ep, const double *eMeasure, 
        const int nQuad5, const double *W5, const double *Lambda5p, 
        const double *Up, double *ugu)
{
    int D = (d+1)*(d+2)/2;
    // convert pointer to TensorAccessor
    int esize[] = {M,D}; int estride[] = {D,1};
    TensorAccessor<const int,2> e(ep,esize,estride);
    int Esize[] = {M,d+1,d}; int Estride[] = {(d+1)*d,d,1};
    TensorAccessor<const double,3> E(Ep,Esize,Estride);
    int Lambda5size[] = {nQuad5,d+1}; int Lambda5stride[] = {d+1,1};
    TensorAccessor<const double,2> Lambda5(Lambda5p,Lambda5size,Lambda5stride);

    for (int i=0; i<d*(N+NE); ++i)
        ugu[i] = 0; // initialize all elements as 0

    // coefficients derived from test function in $e_k$
    for (int k=0; k<M; ++k)
    {
        Tensor<double,2> UeTensor({D,d});
        TensorAccessor<double,2> Ue = UeTensor.accessor();
        const int *ek = e[k].data();
        for (int j=0; j<D; ++j) // update Ue
            for (int l=0; l<d; ++l)
                Ue[j][l] = Up[l*(N+NE)+ek[j]];

        Tensor<double,2> Gamma5Tensor({nQuad5,D});
        TensorAccessor<double,2> Gamma5 = Gamma5Tensor.accessor();
        CalculateGamma(d, D, nQuad5, Lambda5.ConstAccessor(), Gamma5);

        Tensor<double,2> U5Tensor({nQuad5,d});
        TensorAccessor<double,2> U5 = U5Tensor.accessor();
        CalculateU(d, D, nQuad5, Gamma5.ConstAccessor(), Ue.ConstAccessor(), U5);

        Tensor<double,3> Theta5Tensor({nQuad5,D,d});
        TensorAccessor<double,3> Theta5 = Theta5Tensor.accessor();
        CalculateTheta(d, E[k], nQuad5, Lambda5.ConstAccessor(), Theta5);

        Tensor<double,3> GU5Tensor({nQuad5,d,d});
        TensorAccessor<double,3> GU5 = GU5Tensor.accessor();
        CalculateGU(d, D, nQuad5, Theta5.ConstAccessor(), Ue.ConstAccessor(), GU5);

        Tensor<double,2> UGU5Tensor({nQuad5,d});
        TensorAccessor<double,2> UGU5 = UGU5Tensor.accessor();
        CalculateUGU(d, nQuad5, U5.ConstAccessor(), GU5.ConstAccessor(), UGU5);

        // Tensor<double,1> TrGU5Tensor({nQuad5});
        // TensorAccessor<double,1> TrGU5 = TrGU5Tensor.accessor();
        // CalculateTrGU(d, nQuad5, GU5.ConstAccessor(), TrGU5);

        
        for (int j0=0; j0<D; ++j0)
        {
            for (int l=0; l<d; ++l) // $v^{j0,l}$, row=l*(N+NE)+ek[j0]
            {
                int row = l*(N+NE)+ek[j0];
                for (int i=0; i<nQuad5; ++i) // right hand items
                    ugu[row] += eMeasure[k]*W5[i]*Gamma5[i][j0]*UGU5[i][l];
            }
        }
    }
    return 0;
}
