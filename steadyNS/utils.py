import numpy as np
from scipy.sparse.linalg.interface import IdentityOperator

def CG(A,b,x0=None,tol=1e-5,maxiter=20,M=None,callback=None):
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
    return x

