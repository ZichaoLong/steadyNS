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

int _switchEdgeNode(const int L, int *Edge)
{
#pragma omp parallel for schedule(static)
    for (int i=0; i<L; ++i)
        if (Edge[2*i]>Edge[2*i+1])
            std::swap(Edge[2*i],Edge[2*i+1]);
    return 0;
}

int _updateEdgeTags(const int d, const int N, const int NE, const int *Edge, 
        const int *B, const double *coord, int *Bedge)
{
    int tag1,tag2;
#pragma omp parallel for schedule(static) private(tag1,tag2)
    for (int i=0; i<NE; ++i)
    {
        tag1 = B[Edge[2*i]];
        tag2 = B[Edge[2*i+1]];
        if (tag1==0 || tag2==0)
            Bedge[i] = 0;
        else if (tag1==tag2)
            Bedge[i] = tag1;
        else if (tag1<4 && tag2<4)
        {
            double tmp = coord[d*Edge[2*i]]-coord[d*Edge[2*i+1]];
            tmp = tmp*tmp;
            if (tmp<1e-16)
            {
                if (tag1<3 && tag2==3)
                    Bedge[i] = tag1;
                else if (tag1==3 && tag2<3)
                    Bedge[i] = tag2;
                else
                {
                    cout << "Encounter unknown node tags. exit(0)" << endl;
                    exit(0);
                }
            }
            else
            {
                Bedge[i] = 0;
            }
        }
        else
        {
            cout << "One of the holes is too close to wall! exit(0);" << endl;
            exit(0);
        }
    }
    return 0;
}
