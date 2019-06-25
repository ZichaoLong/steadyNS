#!/usr/bin/env python
# -*- coding: utf-8 -*-
#%%
import pyamg
from pyamg.aggregation import smoothed_aggregation_solver
ml0 = smoothed_aggregation_solver(C_sparse)
MM0 = ml0.aspreconditioner(cycle='V')
ml1 = smoothed_aggregation_solver(C_sparse.transpose().tocsr())
MM1 = ml1.aspreconditioner(cycle='V')
rhi = steadyNS.steadyNS.RHI(UP,d,M,N,NE,B,e,E,eMeasure)
rhinew = C_sparse.transpose()@(MM1@(MM0@rhi))
def linop(x):
    return C_sparse.transpose()@(MM1@(MM0@(C_sparse@x)))
A = sp.sparse.linalg.LinearOperator(C_sparse.shape, linop)
def callback(r):
    print(np.linalg.norm(r))
UPtmp,info = sp.sparse.linalg.cg(A,rhinew,callback=callback,maxiter=100)
#%%
import scipy.optimize
def f(x):
    return np.linalg.norm(MM0@(C_sparse@x-rhi))**2/2
def fprime(x):
    return C_sparse.transpose()@(MM1@(MM0@(C_sparse@x-rhi)))
xopt,fopt,dopt = sp.optimize.fmin_l_bfgs_b(f,np.random.randn(rhi.shape[0]),fprime,iprint=50)

#%%


