# distutils: language = c++
# cython: language_level=3

cdef extern from "steadyNS.h":
    int _CsrMulVec(const int M, const int N, const int nnz, 
            const int *IA, const int *JA, const double *data, 
            const double *x, double *y)
    int _InterpP2ToUniformGrid(const double dx,
            const int xidxrange, const int yidxrange, const int zidxrange,
            const int M, const int N, const int NE,
            const int *ep , const double *basisp, const double *coordAllp,
            const double *Up, double *uniformUp)
    int _InterpP1ToUniformGrid(const double dx,
            const int xidxrange, const int yidxrange, const int zidxrange,
            const int M, const int N, const int NE,
            const int *ep , const double *basisp, const double *coordAllp,
            const double *Pp, double *uniformPp)

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

def BarycentricBasis(d,M,e,coord):
    E = np.empty((M,d+1,d+1),dtype=float)
    E[:,0,:] = 1
    for i in range(d+1):
        E[:,1:,i] = coord[e[:,i]]
    E = np.linalg.inv(E)
    return E

def InterpP2ToUniformGrid(dx,maxx,maxy,maxz,U0,M,N,NE,e,coordAll):
    """
    Interpolation from P2 finite element to uniform grid
    This function is only for 3d case
    """
    e = np.ascontiguousarray(e)
    cdef int[::1] e_memview = e.reshape(-1)
    E = BarycentricBasis(3,M,e,coordAll)
    cdef double[::1] E_memview = E.reshape(-1)
    coordAll = np.ascontiguousarray(coordAll)
    cdef double[::1] coordAll_memview = coordAll.reshape(-1)
    U = np.ascontiguousarray(U0)
    cdef double[::1] U_memview = U.reshape(-1)
    xidxrange,yidxrange,zidxrange = \
            int(np.floor(maxx/dx)),int(np.floor(maxy/dx)),int(np.floor(maxz/dx))
    uniformU = np.zeros((xidxrange,yidxrange,zidxrange))
    cdef double[::1] uniformU_memview = uniformU.reshape(-1)
    _InterpP2ToUniformGrid(dx,xidxrange,yidxrange,zidxrange,
            M,N,NE,&e_memview[0],&E_memview[0],&coordAll_memview[0],
            &U_memview[0],&uniformU_memview[0])
    return uniformU

def InterpP1ToUniformGrid(dx,maxx,maxy,maxz,P0,M,N,NE,e,coordAll):
    """
    Interpolation from P1 finite element to uniform grid
    This function is only for 3d case
    """
    e = np.ascontiguousarray(e)
    cdef int[::1] e_memview = e.reshape(-1)
    E = BarycentricBasis(3,M,e,coordAll)
    cdef double[::1] E_memview = E.reshape(-1)
    coordAll = np.ascontiguousarray(coordAll)
    cdef double[::1] coordAll_memview = coordAll.reshape(-1)
    P = np.ascontiguousarray(P0)
    cdef double[::1] P_memview = P.reshape(-1)
    xidxrange,yidxrange,zidxrange = \
            int(np.floor(maxx/dx)),int(np.floor(maxy/dx)),int(np.floor(maxz/dx))
    uniformP = np.zeros((xidxrange,yidxrange,zidxrange))
    cdef double[::1] uniformP_memview = uniformP.reshape(-1)
    _InterpP1ToUniformGrid(dx,xidxrange,yidxrange,zidxrange,
            M,N,NE,&e_memview[0],&E_memview[0],&coordAll_memview[0],
            &P_memview[0],&uniformP_memview[0])
    return uniformP
