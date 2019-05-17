/*************************************************************************
  > File Name: steadyNSCPP.h
  > Author: Long Zichao
  > Mail: zlong@pku.edu.cn
  > Created Time: 2019-05-16
 ************************************************************************/

#ifndef STEADYNSCPP_H
#define STEADYNSCPP_H

int _reduceP(const int N, long *P);
int _mergePeriodNodes(const int d, const int M,
        const long *B, const long *P, long *ep);

#endif // STEADYNSCPP_H
