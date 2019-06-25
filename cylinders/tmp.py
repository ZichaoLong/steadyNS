#!/usr/bin/env python
# -*- coding: utf-8 -*-
#%%
import scipy.optimize
import time
startt = time.time()
def f(x):
    return np.linalg.norm(C@x-rhi)**2/2
def fprime(x):
    return C.transpose()@(C@x-rhi)
xopt,fopt,dopt = sp.optimize.fmin_l_bfgs_b(f,UP,fprime=fprime,m=10,iprint=90,pgtol=1e-8,factr=1e6,maxfun=1e6,maxiter=1e6)
print("elapsed time: ", time.time()-startt)
#%%
import time
startt = time.time()
def linop(x):
    return C.transpose()@(C@x)
MM = sp.sparse.linalg.LinearOperator(C.shape,linop)
x,info = sp.sparse.linalg.minres(MM,C.transpose()@rhi,x0=UP,maxiter=1e6,tol=1e-16,show=True)
print("elapsed time: ", time.time()-startt)

#%%


