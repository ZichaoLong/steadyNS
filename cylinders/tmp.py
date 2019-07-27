#!/usr/bin/env python
# -*- coding: utf-8 -*-
#%% solve newton linear system with iterative methods
Bd = np.concatenate([B,]*d,axis=0)
U0 = UP0[:d*DN].reshape(d,DN)
U0 = steadyNS.steadyNS.EmbedU(d,N,NE,B,U0)
ugu,UplusGU,UGUplus = steadyNS.steadyNS.UGU(U0,d,M,N,NE,B,e,E,eMeasure)
UPrhi = np.zeros_like(UP0)
UPrhi[:d*DN] = ugu[:,B==0].reshape(-1)
UPrhi[:d*DN] -= nu*URHIAdd.reshape(-1)
UplusGU = UplusGU[Bd==0]
UGUplus = UGUplus[Bd==0]
NewtonURHIAdd = np.zeros(d*DN)
tmp = steadyNS.steadyNS.ReturnU(d,N,NE,B)
UPrhi[:d*DN] -= UplusGU@tmp.reshape(-1)
UPrhi[:d*DN] -= UGUplus@tmp.reshape(-1)
UplusGU = UplusGU[:,Bd==0]
UGUplus = UGUplus[:,Bd==0]
UPrhi[d*DN:] -= PRHIAdd
BigPoisson = sp.sparse.bmat([[C,None],[None,C]])
BigC = nu*BigPoisson+UplusGU+UGUplus
BigC0 = sp.sparse.bmat([[C0[0],C0[1]],])
BigNewton = sp.sparse.bmat([[BigC,BigC0.transpose().tocsr()],[BigC0,None]],format='csr')
k = 0
NUMCALL = 0
def callback(xk): 
    global k
    global ConvergenceInfo
    # ConvergenceInfo.append(np.linalg.norm(xk))
    ConvergenceInfo.append(np.linalg.norm(steadyNS.utils.CsrMulVec(BigNewton,xk)-UPrhi))
    if k%50==0:
        pass
        # print("IterNum: ",k," ResNorm: ",np.linalg.norm(steadyNS.utils.CsrMulVec(BigNewton,xk)-UPrhi)) # for lgmres,bicg,bicgstab
        # print("IterNum: ",k," ResNorm: ",np.linalg.norm(xk)) # for gmres, it is very slow
    k += 1
def matvec(x):
    global NUMCALL
    if NUMCALL%50==0:
        pass
        # print(NUMCALL)
    NUMCALL += 1
    return steadyNS.utils.CsrMulVec(BigNewton,x)
BigNewtonT = BigNewton.transpose().tocsr()
testNewton = sp.sparse.linalg.LinearOperator(
        shape=BigNewton.shape,
        # matvec=lambda x:steadyNS.utils.CsrMulVec(BigNewton,x),
        matvec=matvec,
        rmatvec=lambda x:steadyNS.utils.CsrMulVec(BigNewtonT,x)
        )
# ## precondition method 0
# BigNewtonML = smoothed_aggregation_solver(BigNewton,symmetry='nonsymmetric',strength=None)
# BigNewtonMM = BigNewtonML.aspreconditioner(cycle='V')
## precondition method 1
BigCML = smoothed_aggregation_solver(BigC,symmetry='nonsymmetric',
        strength=('predefined',{'C':BigPoisson}))
BigCMM = BigCML.aspreconditioner(cycle='V')
def BigNewtonMMOP(x):
    y = x.copy()
    y[:d*DN] = BigCMM@x[:d*DN]
    return y
BigNewtonMM = sp.sparse.linalg.LinearOperator(shape=BigNewton.shape,matvec=BigNewtonMMOP)
 
ConvergenceInfo = []
startt = time.time()
# UP,info = sp.sparse.linalg.gmres(testNewton,UPrhi,callback=callback,tol=1e-10,atol=1e-10,maxiter=2000,M=BigNewtonMM)
# UP,info = sp.sparse.linalg.lgmres(testNewton,UPrhi,callback=callback,tol=1e-10,atol=1e-10,maxiter=2000,M=BigNewtonMM)
UP,info = sp.sparse.linalg.lgmres(testNewton,UPrhi,callback=callback,tol=1e-10,atol=1e-10,maxiter=2000)
# UP,info = sp.sparse.linalg.bicg(testNewton,UPrhi,callback=callback,tol=1e-10,atol=1e-10,maxiter=20000)
print(time.time()-startt)

#%%
# UP0 = np.zeros(d*DN+M-1)
# nu = 0.1 # change viscosity here
Bd = np.concatenate([B,]*d,axis=0)
def NEWTONITE(UP0,tol):
    global k
    U0 = UP0[:d*DN].reshape(d,DN)
    U0 = steadyNS.steadyNS.EmbedU(d,N,NE,B,U0)
    ugu,UplusGU,UGUplus = steadyNS.steadyNS.UGU(U0,d,M,N,NE,B,e,E,eMeasure)
    UPrhi = np.zeros_like(UP0)
    UPrhi[:d*DN] = ugu[:,B==0].reshape(-1)
    UPrhi[:d*DN] -= nu*URHIAdd.reshape(-1)
    UplusGU = UplusGU[Bd==0]
    UGUplus = UGUplus[Bd==0]
    NewtonURHIAdd = np.zeros(d*DN)
    tmp = steadyNS.steadyNS.ReturnU(d,N,NE,B)
    UPrhi[:d*DN] -= UplusGU@tmp.reshape(-1)
    UPrhi[:d*DN] -= UGUplus@tmp.reshape(-1)
    UplusGU = UplusGU[:,Bd==0]
    UGUplus = UGUplus[:,Bd==0]
    UPrhi[d*DN:] -= PRHIAdd
    BigPoisson = sp.sparse.bmat([[C,None],[None,C]])
    BigC = nu*BigPoisson+UplusGU+UGUplus
    BigC0 = sp.sparse.bmat([[C0[0],C0[1]],])
    BigNewtonMatrix = sp.sparse.bmat([[BigC,BigC0.transpose().tocsr()],[BigC0,None]],format='csr')
    BigNewtonMatrixT = BigNewtonMatrix.transpose().tocsr()
    BigNewton = sp.sparse.linalg.LinearOperator(shape=BigNewtonMatrix.shape,
            matvec=lambda x:steadyNS.utils.CsrMulVec(BigNewtonMatrix,x),
            rmatvec=lambda x:steadyNS.utils.CsrMulVec(BigNewtonMatrixT,x))
    BigCML = smoothed_aggregation_solver(BigC,symmetry='nonsymmetric',
            strength=('predefined',{'C':BigPoisson}))
    BigCMM = BigCML.aspreconditioner(cycle='V')
    def BigNewtonMMOP(x):
        y = x.copy()
        y[:d*DN] = BigCMM@x[:d*DN]
        return y
    BigNewtonMM = sp.sparse.linalg.LinearOperator(shape=BigNewton.shape,matvec=BigNewtonMMOP)
    def callback(xk):
        global k
        if k%3==0:
            print("lgmres IterNum: ", k, ' ResNorm: ', np.linalg.norm(BigNewton@xk-UPrhi))
        k += 1
    k = 0
    UP,info = sp.sparse.linalg.lgmres(BigNewton,UPrhi,tol=tol,atol=tol,maxiter=1000,
            callback=callback,M=BigNewtonMM)
    # UP = sp.sparse.linalg.spsolve(BigNewton, UPrhi)
    print("Newton Precision: ", np.linalg.norm(BigNewton@UP-UPrhi))
    return UP
tol = [1e-5,1e-6,1e-6,1e-6,1e-7,1e-8,1e-9]+[1e-10,]*100
startt = time.time()
for i in range(5):
    UP1 = NEWTONITE(UP0,tol[i])
    print("Newton Iter Convergence: ", np.linalg.norm(UP0-UP1))
    UP0 = UP1
print("elapsed time: ", time.time()-startt)
U = UP0[:d*DN].reshape(d,DN)
U = steadyNS.steadyNS.EmbedU(d,N,NE,B,U)
P = UP0[d*DN:]
# P = np.concatenate([UP0[d*DN:],np.zeros(1)])
#%%
