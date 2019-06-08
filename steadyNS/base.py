import numpy as np
from . import QuadPts

def Gamma(d,order):
    W,Lambda = QuadPts.quadpts(d,order)
    nQuad = W.size
    G = np.zeros((nQuad,(d+1)*(d+2)//2))
    for j in range(d+1):
        G[:,j] = 2*Lambda[:,j]*(Lambda[:,j]-0.5)
    for j1 in range(1,d+1):
        for j2 in range(j1):
            j += 1
            G[:,j] = 4*Lambda[:,j1]*Lambda[:,j2]
    assert(j==(d+1)*(d+2)//2)
    return G

def VU(d,order):
    W,Lambda = QuadPts.quadpts(d,order)
    nQuad = W.size
    G = Gamma(d,order)
    D = (d+1)*(d+2)//2
    VU = W[:,np.newaxis,np.newaxis]*G[:,:,np.newaxis]*G[:,np.newaxis,:]
    return VU
