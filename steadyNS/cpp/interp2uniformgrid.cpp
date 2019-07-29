/*************************************************************************
  > File Name: interp2uniformgrid.cpp
  > Author: Long Zichao
  > Mail: zlong@pku.edu.cn
  > Created Time: 2019-07-29
 ************************************************************************/

#include<iostream>
#include<cmath>
#include "steadyNS.h"
#include "ASTen/TensorAccessor.h"
#include "ASTen/Tensor.h"
using std::cout; using std::endl; using std::ends;

int IdxRangeOfEle(const int d, const double dx, const int *ek, 
        const TensorAccessor<const double,2> &coordAll, 
        int *idxmin, int *idxmax)
{
    double coordmin[] = {0,0,0};
    double coordmax[] = {0,0,0};
    for (int l=0; l<d; ++l)
    {
        coordmin[l] = coordAll[ek[0]][l];
        coordmax[l] = coordAll[ek[0]][l];
    }
    double tmp = 0;
    for (int j=1; j<d+1; ++j)
        for (int l=0; l<d; ++l)
        {
            tmp = coordAll[ek[j]][l];
            if (coordmin[l]>tmp)
                coordmin[l] = tmp;
            if (coordmax[l]<tmp)
                coordmax[l] = tmp;
        }
    for (int l=0; l<d; ++l)
    {
        idxmin[l] = int(std::ceil(coordmin[l]/dx));
        idxmax[l] = int(std::floor(coordmax[l]/dx));
    }
    return 0;
}

bool barycentric(const int d, const double *basis, 
        const double *point, double *lambda)
{
    bool isinelement = true;
    for (int j0=0; j0<d+1; ++j0)
    {
        const double *basisrow = basis+j0*(d+1);
        lambda[j0] = basisrow[0];
        for (int j1=1; j1<d+1; ++j1)
            lambda[j0] += basisrow[j1]*point[j1-1];
        if (lambda[j0]<0)
            isinelement = false;
    }
    return isinelement;
}

double _InterpP2(const int d, const double *Uk, const double *lambda)
{
    double u=0;
    for (int j=0; j<d+1; ++j)
        u += Uk[j]*2*lambda[j]*(lambda[j]-0.5);
    for (int j1=0; j1<d+1; ++j1)
        for (int j2=0; j2<j1; ++j2)
            u += Uk[d+1+j1*(j1-1)/2+j2]*4*lambda[j1]*lambda[j2];
    return u;
}

double _InterpP1(const int d, const double *Pk, const double *lambda)
{
    double p=0;
    for (int j=0; j<d+1; ++j)
        p += Pk[j]*lambda[j];
    return p;
}

int _InterpP2ToUniformGrid(const double dx,
        const int xidxrange, const int yidxrange, const int zidxrange,
        const int M, const int N, const int NE,
        const int *ep , const double *basisp, const double *coordAllp,
        const double *Up, double *uniformUp)
{
    const int d = 3;
    int D = (d+1)*(d+2)/2;
    // convert pointer to TensorAccessor
    int esize[] = {M,D}; int estride[] = {D,1};
    TensorAccessor<const int,2> e(ep,esize,estride);
    int basissize[] = {M,d+1,d+1};
    int basisstride[] = {(d+1)*(d+1),d+1,1};
    TensorAccessor<const double,3> basis(basisp,basissize,basisstride);
    int coordAllsize[] = {N+NE,d}; int coordAllstride[] = {d,1};
    TensorAccessor<const double,2> coordAll(coordAllp, coordAllsize, coordAllstride);
    int uniformUsize[] = {xidxrange,yidxrange,zidxrange};
    int uniformUstride[] = {yidxrange*zidxrange,zidxrange,1};
    TensorAccessor<double,3> uniformU(uniformUp,uniformUsize,uniformUstride);

#pragma omp parallel for schedule(dynamic)
    for (int k=0; k<M; ++k)
    {
        const int *ek = e[k].data();
        double Uk[10];
        for (int j=0; j<D; ++j)
            Uk[j] = Up[ek[j]];
        int idxmin[] = {0,0,0};
        int idxmax[] = {0,0,0};
        IdxRangeOfEle(d, dx, ek, coordAll, idxmin, idxmax);
        double point[] = {0,0,0};
        double lambda[] = {0,0,0,0};
        const double *basisk = basis[k].data();
        for (int idxx=idxmin[0]; idxx<idxmax[0]+1; ++idxx)
        {
            if (idxx<0 || idxx>=xidxrange)
                continue;
            point[0] = idxx*dx;
            for (int idxy=idxmin[1]; idxy<idxmax[1]+1; ++idxy)
            {
                if (idxy<0 || idxy>=yidxrange)
                    continue;
                point[1] = idxy*dx;
                for (int idxz=idxmin[2]; idxz<idxmax[2]+1; ++idxz)
                {
                    if (idxz<0 || idxz>=zidxrange)
                        continue;
                    point[2] = idxz*dx;
                    if (barycentric(d, basisk, point, lambda))
                        uniformU[idxx][idxy][idxz] = _InterpP2(d, Uk, lambda);
                }
            }
        }
    }
    return 0;
}
int _InterpP1ToUniformGrid(const double dx,
        const int xidxrange, const int yidxrange, const int zidxrange,
        const int M, const int N, const int NE,
        const int *ep , const double *basisp, const double *coordAllp,
        const double *Pp, double *uniformPp)
{
    const int d = 3;
    int D = (d+1)*(d+2)/2;
    // convert pointer to TensorAccessor
    int esize[] = {M,D}; int estride[] = {D,1};
    TensorAccessor<const int,2> e(ep,esize,estride);
    int basissize[] = {M,d+1,d+1};
    int basisstride[] = {(d+1)*(d+1),d+1,1};
    TensorAccessor<const double,3> basis(basisp,basissize,basisstride);
    int coordAllsize[] = {N+NE,d}; int coordAllstride[] = {d,1};
    TensorAccessor<const double,2> coordAll(coordAllp, coordAllsize, coordAllstride);
    int uniformPsize[] = {xidxrange,yidxrange,zidxrange};
    int uniformPstride[] = {yidxrange*zidxrange,zidxrange,1};
    TensorAccessor<double,3> uniformP(uniformPp,uniformPsize,uniformPstride);

#pragma omp parallel for schedule(dynamic)
    for (int k=0; k<M; ++k)
    {
        const int *ek = e[k].data();
        double Pk[4];
        for (int j=0; j<d+1; ++j)
            Pk[j] = Pp[ek[j]];
        int idxmin[] = {0,0,0};
        int idxmax[] = {0,0,0};
        IdxRangeOfEle(d, dx, ek, coordAll, idxmin, idxmax);
        double point[] = {0,0,0};
        double lambda[] = {0,0,0,0};
        const double *basisk = basis[k].data();
        for (int idxx=idxmin[0]; idxx<idxmax[0]+1; ++idxx)
        {
            if (idxx<0 || idxx>=xidxrange)
                continue;
            point[0] = idxx*dx;
            for (int idxy=idxmin[1]; idxy<idxmax[1]+1; ++idxy)
            {
                if (idxy<0 || idxy>=yidxrange)
                    continue;
                point[1] = idxy*dx;
                for (int idxz=idxmin[2]; idxz<idxmax[2]+1; ++idxz)
                {
                    if (idxz<0 || idxz>=zidxrange)
                        continue;
                    point[2] = idxz*dx;
                    if (barycentric(d, basisk, point, lambda))
                        uniformP[idxx][idxy][idxz] = _InterpP1(d, Pk, lambda);
                }
            }
        }
    }
    return 0;
}
