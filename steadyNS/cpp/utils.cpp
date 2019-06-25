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

int UpdateStiffMatTheta1Sum(const int d, const int D, const TensorAccessor<const double,2> &Ek,
        const int nQuad1, const double *W1, const TensorAccessor<const double,2> &Lambda1,
        TensorAccessor<double,2> &Theta1Sum)
{
    Tensor<double,3> Theta1Tensor({nQuad1,D,d});
    TensorAccessor<double,3> Theta1 = Theta1Tensor.accessor();
    CalculateTheta(d,Ek,nQuad1,Lambda1,Theta1);
    // update Theta1Sum
    for (int j=0; j<D; ++j)
        for (int l=0; l<d; ++l)
        {
            Theta1Sum[j][l] = 0;
            for (int i=0; i<nQuad1; ++i)
                Theta1Sum[j][l] += W1[i]*Theta1[i][j][l];
        }
    return 0;
}

int UpdateStiffMatTheta2Sum(const int d, const int D, const TensorAccessor<const double,2> &Ek, 
        const int nQuad2, const double *W2, const TensorAccessor<const double,2> &Lambda2, 
        TensorAccessor<double,2> &Theta2Sum)
{
    Tensor<double,3> Theta2Tensor({nQuad2,D,d});
    TensorAccessor<double,3> Theta2 = Theta2Tensor.accessor();
    CalculateTheta(d,Ek,nQuad2,Lambda2,Theta2);
    // update Theta2Sum
    for (int j0=0; j0<D; ++j0)
        for (int j1=0; j1<D; ++j1)
        {
            Theta2Sum[j0][j1] = 0;
            for (int i=0; i<nQuad2; ++i)
                for (int l=0; l<d; ++l)
                    Theta2Sum[j0][j1] += W2[i]*Theta2[i][j0][l]*Theta2[i][j1][l];
        }
    return 0;
}

int CalculateTheta(const int d, const TensorAccessor<const double,2> &Ek,
        const int nQuad, const TensorAccessor<const double,2> &Lambda,
        TensorAccessor<double,3> &Theta)
{
    for (int i=0; i<nQuad; ++i)
    {
        for (int j=0; j<d+1; ++j)
            for (int l=0; l<d; ++l)
                Theta[i][j][l] = 4*Lambda[i][j]*Ek[j][l]-Ek[j][l];
        for (int j1=0; j1<d+1; ++j1)
            for (int j2=0; j2<j1; ++j2)
                for (int l=0; l<d; ++l)
                    Theta[i][d+1+j1*(j1-1)/2+j2][l] = 
                        4*(Lambda[i][j1]*Ek[j2][l]+Lambda[i][j2]*Ek[j1][l]);
    }
    return 0;
}

int CalculateGamma(const int d, const int D, const int nQuad, 
        const TensorAccessor<const double,2> &Lambda, 
        TensorAccessor<double,2> Gamma)
{
    for (int j=0; j<d+1; ++j)
        for (int i=0; i<nQuad; ++i)
            Gamma[i][j] = 2*Lambda[i][j]*(Lambda[i][j]-0.5);
    for (int i=0; i<nQuad; ++i)
        for (int j1=0; j1<d+1; ++j1)
            for (int j2=0; j2<j1; ++j2)
                Gamma[i][d+1+j1*(j1-1)/2+j2] = 4*Lambda[i][j1]*Lambda[i][j2];
    return 0;
}

int CalculateU(const int d, const int D, const int nQuad, 
        const TensorAccessor<const double,2> &Gamma, 
        const TensorAccessor<const double,2> &Ue, 
        TensorAccessor<double,2> &U)
{
    for (int i=0; i<nQuad; ++i)
        for (int l=0; l<d; ++l)
        {
            U[i][l] = 0;
            for (int j=0; j<D; ++j)
                U[i][l] += Gamma[i][j]*Ue[j][l];
        }
    return 0;
}

int CalculateGU(const int d, const int D, const int nQuad, 
        const TensorAccessor<const double,3> &Theta, 
        const TensorAccessor<const double,2> &Ue, 
        TensorAccessor<double,3> &GU)
{
    for (int i=0; i<nQuad; ++i)
        for (int l=0; l<d; ++l)
            for (int l1=0; l1<d; ++l1)
            {
                GU[i][l][l1] = 0;
                for (int j=0; j<D; ++j)
                    GU[i][l][l1] += Theta[i][j][l1]*Ue[j][l];
            }
    return 0;
}

int CalculateUGU(const int d, const int nQuad, 
        const TensorAccessor<const double,2> &U, 
        const TensorAccessor<const double,3> &GU, 
        TensorAccessor<double,2> &UGU)
{
    for (int i=0; i<nQuad; ++i)
        for (int l=0; l<d; ++l)
        {
            UGU[i][l] = 0;
            for (int l1=0; l1<d; ++l1)
                UGU[i][l] += U[i][l1]*GU[i][l][l1];
        }
    return 0;
}

int CalculateTrGU(const int d, const int nQuad, 
        const TensorAccessor<const double,3> &GU, 
        TensorAccessor<double,1> &TrGU)
{
    for (int i=0; i<nQuad; ++i)
    {
        TrGU[i] = 0;
        for (int l=0; l<d; ++l)
            TrGU[i] += GU[i][l][l];
    }
    return 0;
}

