/*************************************************************************
  > File Name: utils.cpp
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

int _StiffMatOO_Boundary(const int d, const int N, const int NE, 
        const int *B, const int *P, 
        int &idx, int *I, int *J, double *data)
{
    // equations derived from boundary conditions
    for (int i=0; i<N+NE; ++i)
    {
        if (B[i]==3) // periodic nodes i; 
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

int CalculateTheta(const int d, const TensorAccessor<const double,2> &Ek,
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

int UpdateStiffMatTheta1Sum(const int d, const int D, const TensorAccessor<const double,2> &Ek,
        const int nQuad1, const double *W1, TensorAccessor<const double,2> &Lambda1,
        TensorAccessor<double,2> &Theta1Sum)
{
    Tensor<double,3> Theta1Tensor({nQuad1,D,d});
    TensorAccessor<double,3> Theta1 = Theta1Tensor.accessor();
    CalculateTheta(d,Ek,nQuad1,Lambda1,Theta1);
    double tmp;
    // update Theta1Sum
    for (int j=0; j<D; ++j)
        for (int l=0; l<d; ++l)
        {
            tmp = 0;
            for (int i=0; i<nQuad1; ++i)
                tmp += W1[i]*Theta1[i][j][l];
            Theta1Sum[j][l] = tmp;
        }
    return 0;
}

int UpdateStiffMatTheta2Sum(const int d, const int D, const TensorAccessor<const double,2> &Ek, 
        const int nQuad2, const double *W2, TensorAccessor<const double,2> &Lambda2, 
        TensorAccessor<double,2> &Theta2Sum)
{
    Tensor<double,3> Theta2Tensor({nQuad2,D,d});
    TensorAccessor<double,3> Theta2 = Theta2Tensor.accessor();
    CalculateTheta(d,Ek,nQuad2,Lambda2,Theta2);
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

