/*************************************************************************
  > File Name: mesh_extension.cpp
  > Author: Long Zichao
  > Mail: zlong@pku.edu.cn
  > Created Time: 2019-05-16
 ************************************************************************/

#include<iostream>
#include<algorithm>
#include<cmath>
#include "steadyNS.h"
#include "ASTen/TensorAccessor.h"
using std::cout; using std::endl; using std::ends;

int _reduceP(const int N, int *P)
{
    for (int i=0; i<N; ++i)
    {
        if (P[i]!=-1)
            if (P[P[i]]!=-1) P[i] = P[P[i]];
    }
    return 0;
}

int _mergePeriodNodes(const int d, const int M,
        const int *B, const int *P, int *ep)
{
    int esize[] = {M,d+1};
    int estride[] = {d+1,1};
    TensorAccessor<int,2> e(ep,esize,estride);
#pragma omp parallel for schedule(static)
    for (int k=0; k<M; ++k)
        for (int j=0; j<d+1; ++j)
            if (B[e[k][j]]==3)
                e[k][j] = P[e[k][j]];
    return 0;
}

int _switchEdgeNode(const int L, int *Edge)
{
#pragma omp parallel for schedule(static)
    for (int i=0; i<L; ++i)
        if (Edge[2*i]>Edge[2*i+1])
            std::swap(Edge[2*i],Edge[2*i+1]);
    return 0;
}

int _updateEdgeTags(const int N, const int NE, const int *Edge, 
        const int *B, int *Bedge)
{
    int tag1,tag2;
#pragma omp parallel for schedule(static) private(tag1,tag2)
    for (int i=0; i<NE; ++i)
    {
        tag1 = B[Edge[2*i]];
        tag2 = B[Edge[2*i+1]];
        if (tag1==0 || tag2==0)
            Bedge[i] = 0;
        else if (tag1==-1 || tag2==-1)
            Bedge[i] = -1;
        else if (tag1==tag2)
            Bedge[i] = tag1;
        else 
        {
            cout << "error" << endl;
            exit(0);
        }
    }
    return 0;
}
