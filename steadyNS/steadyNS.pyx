# distutils: language = c++
# cython: language_level=3

import numpy as np
import scipy as sp
import scipy.sparse
from . import QuadPts
from . import poisson

def StiffMat(d,M,N,NE,B,e,E,eMeasure):
    D = (d+1)*(d+2)//2
    C_NUM = M*D*d
    I = np.empty(C_NUM,dtype=np.int32)
    J = np.empty(C_NUM,dtype=np.int32)
    data = np.empty(C_NUM,dtype=float)
    cdef int[::1] I_memview = I;
    cdef int[::1] J_memview = J;
    cdef double[::1] data_memview = data;
    B = np.ascontiguousarray(B)
    e = np.ascontiguousarray(e)
    cdef int[::1] B_memview = B
    cdef int[::1] e_memview = e.reshape(-1)
    E = np.ascontiguousarray(E)
    eMeasure = np.ascontiguousarray(eMeasure)
    cdef double[::1] E_memview = E.reshape(-1)
    cdef double[::1] eMeasure_memview = eMeasure
    W1,Lambda1 = QuadPts.quadpts(d,1)
    cdef double[::1] W1_memview = W1
    cdef double[::1] Lambda1_memview = Lambda1.reshape(-1)
    idx = _StiffMatOO(d, M, N, NE,
            &B_memview[0], &e_memview[0],
            &E_memview[0], &eMeasure_memview[0],
            W1.size, &W1_memview[0], &Lambda1_memview[0], 
            &I_memview[0], &J_memview[0], &data_memview[0])
    print("idx=",idx,", C_NUM=",C_NUM)
    C = []
    for l in range(d):
        C.append(sp.sparse.coo_matrix((data[l::d],(I[l::d],J[l::d])),shape=[M,N+NE]))
    for l in range(d):
        C[l] = C[l].tocsr()
    return C

def TrueUPF(nu,d,N,NE,coordAll,coordEle):
    alpha = 1
    U_true = np.zeros([d,N+NE])
    U_true[0] = alpha*coordAll[:,0]*(coordAll[:,0]-1)*(2*coordAll[:,1]-1)
    U_true[1] = -alpha*coordAll[:,1]*(coordAll[:,1]-1)*(2*coordAll[:,0]-1)
    U0x_true = alpha*(2*coordAll[:,0]-1)*(2*coordAll[:,1]-1)
    U0y_true = 2*alpha*coordAll[:,0]*(coordAll[:,0]-1)
    U1y_true = -U0x_true
    U1x_true = -2*alpha*coordAll[:,1]*(coordAll[:,1]-1)
    P_true = alpha*(2*coordEle[:,0]-1)*(2*coordEle[:,1]-1)
    F = np.zeros([d,N+NE])
    F[0] = (1-nu)*2*alpha*(2*coordAll[:,1]-1)+(U_true[0]*U0x_true+U_true[1]*U0y_true)
    F[1] = (1+nu)*2*alpha*(2*coordAll[:,0]-1)+(U_true[0]*U1x_true+U_true[1]*U1y_true)
    return U_true,P_true,F

def ReturnU(nu,d,N,NE,B,coordAll,coordEle):
    U,_,_ = TrueUPF(nu,d,N,NE,coordAll,coordEle)
    U[:,B==0] = 0
    return U

def EmbedU(nu,d,N,NE,B,U0,coordAll,coordEle):
    U = ReturnU(nu,d,N,NE,B,coordAll,coordEle)
    for l in range(d):
        U[l,B==0] = U0[l]
    return U

def UGU(U0,d,M,N,NE,B,e,E,eMeasure):
    B = np.ascontiguousarray(B)
    e = np.ascontiguousarray(e)
    cdef int[::1] B_memview = B
    cdef int[::1] e_memview = e.reshape(-1)
    E = np.ascontiguousarray(E)
    eMeasure = np.ascontiguousarray(eMeasure)
    cdef double[::1] E_memview = E.reshape(-1)
    cdef double[::1] eMeasure_memview = eMeasure
    W5,Lambda5 = QuadPts.quadpts(d,5)
    cdef double[::1] W5_memview = W5
    cdef double[::1] Lambda5_memview = Lambda5.reshape(-1)
    U = np.ascontiguousarray(U0)
    cdef double[::1] U_memview = U.reshape(-1)
    ugu = np.empty_like(U)
    cdef double[::1] ugu_memview = ugu.reshape(-1)
    C_NUM_UplusGU = _countStiffMatUplusGU(d,M,N,NE,&B_memview[0],&e_memview[0])
    C_NUM_UGUplus = _countStiffMatUGUplus(d,M,N,NE,&B_memview[0],&e_memview[0])
    IUplusGU = np.zeros(C_NUM_UplusGU,dtype=np.int32)
    JUplusGU = np.zeros_like(IUplusGU)
    dataUplusGU = np.zeros(C_NUM_UplusGU,dtype=float)
    IUGUplus = np.zeros(C_NUM_UGUplus,dtype=np.int32)
    JUGUplus = np.zeros_like(IUGUplus)
    dataUGUplus = np.zeros(C_NUM_UGUplus,dtype=float)
    cdef int[::1] IUplusGU_m = IUplusGU
    cdef int[::1] JUplusGU_m = JUplusGU
    cdef int[::1] IUGUplus_m = IUGUplus
    cdef int[::1] JUGUplus_m = JUGUplus
    cdef double[::1] dataUplusGU_m = dataUplusGU
    cdef double[::1] dataUGUplus_m = dataUGUplus
    _UGU(C_NUM_UplusGU, C_NUM_UGUplus, d, M, N, NE, &B_memview[0], &e_memview[0], &E_memview[0], 
            &eMeasure_memview[0], W5.size, &W5_memview[0], &Lambda5_memview[0], 
            &U_memview[0], &ugu_memview[0], 
            &IUplusGU_m[0], &JUplusGU_m[0], &dataUplusGU_m[0],
            &IUGUplus_m[0], &JUGUplus_m[0], &dataUGUplus_m[0])
    UplusGU = sp.sparse.coo_matrix((dataUplusGU,(IUplusGU,JUplusGU)),shape=(d*(N+NE),d*(N+NE)))
    UGUplus = sp.sparse.coo_matrix((dataUGUplus,(IUGUplus,JUGUplus)),shape=(d*(N+NE),d*(N+NE)))
    UplusGU = UplusGU.tocsr()
    UGUplus = UGUplus.tocsr()
    return ugu,UplusGU,UGUplus

def sourceF(F,d,M,N,NE,B,e,E,eMeasure):
    B = np.ascontiguousarray(B)
    e = np.ascontiguousarray(e)
    cdef int[::1] B_memview = B
    cdef int[::1] e_memview = e.reshape(-1)
    E = np.ascontiguousarray(E)
    eMeasure = np.ascontiguousarray(eMeasure)
    cdef double[::1] E_memview = E.reshape(-1)
    cdef double[::1] eMeasure_memview = eMeasure
    W4,Lambda4 = QuadPts.quadpts(d,4)
    cdef double[::1] W4_memview = W4
    cdef double[::1] Lambda4_memview = Lambda4.reshape(-1)
    F = np.ascontiguousarray(F)
    cdef double[::1] F_memview = F.reshape(-1)
    rhi = np.zeros_like(F)
    cdef double[::1] rhi_memview = rhi.reshape(-1)
    _sourceF(d,M,N,NE,&B_memview[0],&e_memview[0],&E_memview[0],
            &eMeasure_memview[0],W4.size,&W4_memview[0],&Lambda4_memview[0],
            &F_memview[0],&rhi_memview[0])
    return rhi
