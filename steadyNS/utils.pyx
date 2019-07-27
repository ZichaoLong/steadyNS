# distutils: language = c++
# cython: language_level=3

cdef extern from "steadyNS.h":
    int _CsrMulVec(const int M, const int N, const int nnz, 
            const int *IA, const int *JA, const double *data, 
            const double *x, double *y)

import numpy as np
from scipy.sparse.linalg.interface import IdentityOperator

def CG(A,b,x0=None,tol=1e-5,maxiter=20,M=None,callback=None):
    """
    use conjugate gradient methods to solve Ax=b, where initial value is x0. 
    Iteration will stop at iteration limit 'maxiter' or relative tolerance condition ||A@xk-b||<tol*||b||. 
    Solution x and iteration number will be returned.
    """
    if x0 is None:
        x0 = np.zeros_like(b)
    x = x0
    k = 0
    r = b-A@x
    bnorm = np.linalg.norm(b)
    def idop(x):
        return x
    if M is None:
        M = IdentityOperator(shape=A.shape,dtype=A.dtype)
    while np.linalg.norm(r)>tol*bnorm and k<maxiter:
        if not callback is None:
            callback(x)
        z = M@r
        k += 1
        if k==1:
            p = z
            rho = np.dot(r,z)
        else:
            rhotilde = rho
            rho = np.dot(r,z)
            beta = rho/rhotilde
            p = z+beta*p
        w = A@p
        alpha = rho/np.dot(p,w)
        x = x+alpha*p
        r = r-alpha*w
    return x,k

def CsrMulVec(A,x):
    M,N = A.shape
    y = np.zeros(M)
    nnz = A.nnz
    assert(type(x)==np.ndarray and x.size==N)
    cdef int[::1] IA = A.indptr
    cdef int[::1] JA = A.indices
    cdef double[::1] data = A.data
    x = np.ascontiguousarray(x)
    cdef double[::1] xp = x
    cdef double[::1] yp = y
    _CsrMulVec(M,N,nnz,&IA[0],&JA[0],&data[0],&xp[0],&yp[0])
    return y
