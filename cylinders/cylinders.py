"""
python cylinders.py caseName clscale x1 y1 r1 x2 y2 r2 ...
"""
#%%
import gmsh
import math
import numpy as np
import scipy as sp
import scipy.sparse.linalg
import sys
import time
import steadyNS
import matplotlib.pyplot as plt

nu = 0.1
d = 2;
maxx = 24;
maxy = 8;
lcar1 = 0.4;
lcar2 = 0.2;
if len(sys.argv[1:])>=1:
    caseName = sys.argv[1]
else:
    caseName = "cylinders"
if len(sys.argv[2:])>=1:
    lcar1 *= float(sys.argv[2])
    lcar2 *= float(sys.argv[2])
argv = sys.argv[3:]
print("characteristic length scale of")
print("box boundary "+str(lcar1))
print("cylinder boundary "+str(lcar2))

model = gmsh.model
factory = model.geo

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 1)

model.add("cylinder")

#%%
# read cylinders configuration 
# if no external configure is available, x=4,y=2,d=1 is used by default
Cylinders = []
if len(argv)<3:
    # Cylinders.append(dict(x=6,y=4,d=1))
    Cylinders.append(dict(x=5.5,y=3.5,d=1))
    Cylinders.append(dict(x=6.5,y=4.5,d=1))
else:
    k = 0
    while len(argv)-k>=3:
        Cylinders.append(dict(x=float(argv[k]),y=float(argv[k+1]),d=float(argv[k+2])))
        k += 3

print(Cylinders)

#%% construct box
factory.addPoint(0,0,0,lcar1,1)
factory.addPoint(maxx,0,0,lcar1,2)
factory.addPoint(maxx,maxy,0,lcar1,3)
factory.addPoint(0,maxy,0,lcar1,4)

factory.addLine(1,2,1)
factory.addLine(2,3,2)
factory.addLine(3,4,3)
factory.addLine(4,1,4)

BoxCurveLoopTag = factory.addCurveLoop([1,2,3,4])
factory.synchronize()

#%% add box physical groups
PhysicalInletNodes = 1
PhysicalOutletNodes = 2
PhysicalFixWall = 3
PhysicalInlet = 1
PhysicalOutlet = 2
model.addPhysicalGroup(dim=0, tags=[1,4], tag=PhysicalInletNodes)
model.setPhysicalName(dim=0, tag=PhysicalInletNodes, name='PhysicalInletNodes')
model.addPhysicalGroup(dim=0, tags=[2,3], tag=PhysicalOutletNodes)
model.setPhysicalName(dim=0, tag=PhysicalOutletNodes, name='PhysicalOutletNodes')
model.addPhysicalGroup(dim=1, tags=[1,3], tag=PhysicalFixWall)
model.setPhysicalName(dim=1, tag=PhysicalFixWall, name='PhysicalFixWall')
model.addPhysicalGroup(dim=1, tags=[4,], tag=PhysicalInlet)
model.setPhysicalName(dim=1, tag=PhysicalInlet, name='PhysicalInlet')
model.addPhysicalGroup(dim=1, tags=[2,], tag=PhysicalOutlet)
model.setPhysicalName(dim=1, tag=PhysicalOutlet, name='PhysicalOutlet')

#%% add cylinders
def addCylinder(x,y,d,clscale):
    r = d/2
    PointTag = []
    PointTag.append(factory.addPoint(x,  y,  0,clscale))
    PointTag.append(factory.addPoint(x+r,y,  0,clscale))
    PointTag.append(factory.addPoint(x,  y+r,0,clscale))
    PointTag.append(factory.addPoint(x-r,y,  0,clscale))
    PointTag.append(factory.addPoint(x,  y-r,0,clscale))
    CircleArcTag = []
    CircleArcTag.append(factory.addCircleArc(PointTag[1],PointTag[0],PointTag[2]))
    CircleArcTag.append(factory.addCircleArc(PointTag[2],PointTag[0],PointTag[3]))
    CircleArcTag.append(factory.addCircleArc(PointTag[3],PointTag[0],PointTag[4]))
    CircleArcTag.append(factory.addCircleArc(PointTag[4],PointTag[0],PointTag[1]))
    CurveLoopTag = factory.addCurveLoop(CircleArcTag)
    return PointTag, CircleArcTag, CurveLoopTag

PointTags = []
CircleArcTags = []
CurveLoopTags = []
for cylinder in Cylinders:
    PointTag, CircleArcTag, CurveLoopTag = addCylinder(cylinder['x'],cylinder['y'],cylinder['d'],lcar2)
    PointTags.append(PointTag)
    CircleArcTags.append(CircleArcTag)
    CurveLoopTags.append(CurveLoopTag)
AllCircleArcTags = []
for CircleArcTag in CircleArcTags:
    AllCircleArcTags += CircleArcTag
PhysicalCylinderBoundary = 4
model.addPhysicalGroup(dim=1, tags=list(AllCircleArcTags), tag=PhysicalCylinderBoundary)
model.setPhysicalName(dim=1, tag=PhysicalCylinderBoundary, name='PhysicalCylinderBoundary')
factory.synchronize()

#%% construct planesurface
PlaneSurfaceTag = factory.addPlaneSurface([BoxCurveLoopTag,]+CurveLoopTags)
WholeDomainTag = PlaneSurfaceTag
PhysicalPlaneSurface = 100
model.addPhysicalGroup(dim=2, tags=[PlaneSurfaceTag], tag=PhysicalPlaneSurface)
model.setPhysicalName(dim=2, tag=PhysicalPlaneSurface, name='PhysicalPlaneSurface')
factory.synchronize()

#%%
model.mesh.generate(2)

# gmsh.write(caseName+'.msh2')

#%% set physical tags
PhysicalWholeDomain = PhysicalPlaneSurface
PhysicalInlet = PhysicalInlet
PhysicalOutlet = PhysicalOutlet
PhysicalHoleBoundary = PhysicalCylinderBoundary
#%% set coordinates and node boundaries
N,coord,B = steadyNS.mesh.P1Nodes(d, 
        PhysicalWholeDomain, PhysicalInlet, PhysicalOutlet, PhysicalFixWall, 
        PhysicalHoleBoundary)
B[B==PhysicalOutletNodes] = 0

#%% set elements
M,e,E,eMeasure = steadyNS.mesh.P1Elements(d, WholeDomainTag, coord, B)

steadyNS.mesh.P1Check(coord,B,e,Cylinders,maxx)

NE,B,e,Edge = steadyNS.mesh.P2Elements(d, B, e, coord)
coordEdge = (coord[Edge[:,0]]+coord[Edge[:,1]])/2
coordAll = np.concatenate([coord,coordEdge],axis=0)
coordEle = np.zeros((M,2))
for l in range(d+1):
    coordEle += coord[e[:,l]]
coordEle /= d+1
assert(B.shape[0]==NE+N)
print("edge number: ", NE)
print("e.shape: ", e.shape)
DN = (B==0).sum()
print("number of free nodes/edges for velocity: ", DN)

#%% set global stiff matrix for poisson equation
D = (d+1)*(d+2)//2
C_NUM = D*D*M
print("non-zero number of C_OO=",C_NUM)
C = steadyNS.poisson.P2StiffMat(d,M,N,NE,e,E,eMeasure)
C_full = C
print("C shape=",C.shape)
print("C nnz=",C.nnz)
C = C[B==0]
C = C[:,B==0]
if C.shape[0]<2000:
    print("condition number of C=",np.linalg.cond(C.todense()))
    Ctmp = C.todense()
    values,vectors = np.linalg.eig(Ctmp)

#%% test poisson solver
import pyamg
from pyamg.aggregation import smoothed_aggregation_solver
PoissonML = smoothed_aggregation_solver(C,symmetry='hermitian',strength='symmetric')
PoissonMM = PoissonML.aspreconditioner(cycle='V')
U = steadyNS.poisson.ReturnU(N,NE,B)
URHIAdd_poisson = C_full@U
URHIAdd_poisson = np.ascontiguousarray(URHIAdd_poisson[B==0])
b = 0.01-URHIAdd_poisson
k = 0
def callback(xk):
    global k
    k += 1
    return print("iter: ", k, "ResNorm: ", np.linalg.norm(C@xk-b))
U,info = sp.sparse.linalg.cg(C,b,tol=1e-10,M=PoissonMM,callback=callback)
U = steadyNS.poisson.EmbedU(N,NE,B,U)

#%% show poisson solution
fig = plt.figure(figsize=(maxx//2+2,maxy//2))
ax = fig.add_subplot(111)
ax.tricontour(coordAll[:,0],coordAll[:,1],U,levels=30,linewidths=0.5,colors='k')
cntr = ax.tricontourf(coordAll[:,0],coordAll[:,1],U,levels=30,cmap="RdBu_r")
fig.colorbar(cntr,ax=ax)

#%% set source F 
CF = steadyNS.steadyNS.sourceF(d,M,N,NE,e,E,eMeasure)
print("CF shape=",CF.shape)
print("CF nnz=",CF.nnz)
CF_full = CF
CF = CF[B==0]
CF = CF[:,B==0]

#%% set pressure part of global stiff matrix
C0 = steadyNS.steadyNS.QGU(d,M,N,NE,e,E,eMeasure)
for l in range(d):
    print("C0[",l,"] shape=",C0[l].shape)
    print("C0[",l,"] nnz=",C0[l].nnz)
C0_full = list(x for x in C0)
for l in range(d):
    C0[l] = C0[l][:,B==0]

#%% set URHIADD, PRHIADD
U = steadyNS.steadyNS.ReturnU(d,N,NE,B)
URHIAdd = np.zeros_like(U)
for l in range(d):
    URHIAdd[l] = C_full@U[l]
URHIAdd = np.ascontiguousarray(URHIAdd[:,B==0])
PRHIAdd = np.zeros(N)
for l in range(d):
    PRHIAdd += C0_full[l]@U[l]

#%% pressure correction method step 0: set linear system
dt = 0.1
nu = nu
PrintInfo = False
# BigC = nu*sp.sparse.bmat([[C,None],[None,C]])
# solveBigStokes = sp.sparse.linalg.splu(BigStokes).solve

STEP0LinearSystem = (CF/dt+nu*C).tocsr()
STEP0ML = smoothed_aggregation_solver(STEP0LinearSystem,symmetry='hermitian',strength='symmetric')
STEP0MM = STEP0ML.aspreconditioner(cycle='V')
STEP0LinearSystemOPFUNC = lambda x:steadyNS.utils.CsrMulVec(STEP0LinearSystem, x)
STEP0LinearSystemOP = sp.sparse.linalg.LinearOperator(shape=STEP0LinearSystem.shape,
        matvec=STEP0LinearSystemOPFUNC, rmatvec=STEP0LinearSystemOPFUNC)
def STEP0(U0):
    global k
    U0 = U0.reshape(d,DN)
    ugu = steadyNS.steadyNS.UGU(steadyNS.steadyNS.EmbedU(d,N,NE,B,U0),
            d,M,N,NE,e,E,eMeasure)
    STEP0rhi = -ugu[:,B==0]
    for l in range(d):
        STEP0rhi[l] += steadyNS.utils.CsrMulVec(CF, U0[l]/dt)
    STEP0rhi -= nu*URHIAdd
    def callback(xk):
        global k
        k += 1
        if PrintInfo:
            print("iter: ", k, "ResNorm: ", np.linalg.norm(STEP0LinearSystemOP@xk-b))
        return 
    Utilde = np.zeros((d,DN))
    tmp = 0
    iternum = 0
    for l in range(d):
        k = 0
        b = STEP0rhi[l]
        Utilde[l],iternumtmp = steadyNS.utils.CG(STEP0LinearSystemOP, b, 
                x0=U0[l], tol=1e-10, maxiter=50, 
                M=STEP0MM, callback=callback)
        tmp +=  np.linalg.norm(STEP0LinearSystemOP@Utilde[l]-STEP0rhi[l])
        iternum += iternumtmp
    print("STEP0 IterNum: ", iternum, "Precision: ", tmp)
    return Utilde

#%% pressure correction method step 1: set linear system
diagCF = CF.diagonal()
diagCF = sp.sparse.dia_matrix((diagCF,0),shape=CF.shape).tocsr()
diagCF.data[:] = 1/diagCF.data
CFOPFUNC = lambda x:steadyNS.utils.CsrMulVec(CF,x)
CFOP = sp.sparse.linalg.LinearOperator(shape=CF.shape,
        matvec=CFOPFUNC, rmatvec=CFOPFUNC)
diagCFOPFUNC = lambda x:steadyNS.utils.CsrMulVec(diagCF,x)
diagCFOP = sp.sparse.linalg.LinearOperator(shape=diagCF.shape,
        matvec=diagCFOPFUNC, rmatvec=diagCFOPFUNC)
CFSolver = lambda x: steadyNS.utils.CG(CFOP, x, 
        tol=1e-10, maxiter=50, M=diagCFOP)[0]
C0transpose = list(C0[l].transpose().tocsr() for l in range(d))
def PLinearSystemOP(x):
    y = np.zeros_like(x)
    for l in range(d):
        y += steadyNS.utils.CsrMulVec(C0[l],
                CFSolver(steadyNS.utils.CsrMulVec(C0transpose[l],x))
                )
    return y
PLinearSystem = sp.sparse.linalg.LinearOperator(shape=[C0[0].shape[0],]*2,matvec=PLinearSystemOP,rmatvec=PLinearSystemOP)
PrePLinearSystem = 0
for l in range(d):
    PrePLinearSystem = PrePLinearSystem+C0[l]@(diagCF@C0[l].transpose())
PrePLinearSystemML = smoothed_aggregation_solver(PrePLinearSystem,symmetry='hermitian',strength='symmetric')
PrePLinearSystemMM = PrePLinearSystemML.aspreconditioner(cycle='V')
def STEP1(Utilde,P0=None):
    global k
    U = Utilde.copy().reshape(d,DN)
    Prhi = 0
    for l in range(d):
        Prhi = Prhi-steadyNS.utils.CsrMulVec(C0[l],U[l])
    Prhi -= PRHIAdd
    Prhi *= 1/dt
    def callback(xk):
        global k
        k += 1
        if PrintInfo:
            print("iter: ", k, "ResNorm: ", np.linalg.norm(PLinearSystem@xk-Prhi))
        return 
    k = 0
    P0 = (np.zeros_like(Prhi) if P0 is None else P0)
    P,iternum = steadyNS.utils.CG(PLinearSystem,Prhi,x0=P0,tol=1e-10,maxiter=50,M=PrePLinearSystemMM,callback=callback)
    print("STEP1 IterNum: ", iternum, "Precision: ", np.linalg.norm(PLinearSystem@P-Prhi))
    for l in range(d):
        U[l] += dt*CFSolver(steadyNS.utils.CsrMulVec(C0transpose[l],P))
    return U,P

#%% test pressure correction method step0, step1
PrintInfo = True
U0 = np.zeros((d,DN))
startt = time.time()
Utilde = STEP0(U0)
print("step0 elapsed time: ", time.time()-startt)
startt = time.time()
U1,P = STEP1(Utilde)
print("step1 elapsed time: ", time.time()-startt)

#%% pressure correction method
U0 = np.zeros((d,DN))
ALLCONVERGE = []
PrintInfo = False
startt = time.time()
for steps in range(2000):
    Utilde = STEP0(U0)
    if steps==0:
        U1,P = STEP1(Utilde)
    else:
        U1,P = STEP1(Utilde,P)
    CONVERGE = np.abs(U1-U0).max()/dt
    ALLCONVERGE.append(CONVERGE)
    DIVERGENCENORM = PRHIAdd
    for l in range(d):
        DIVERGENCENORM = DIVERGENCENORM+C0[l]@U1[l]
    DIVERGENCENORM = np.linalg.norm(DIVERGENCENORM)
    print("ITER: ", steps, "div U1: {:.2e}".format(DIVERGENCENORM), "max(|U1-U0|/dt): {:.2e}".format(CONVERGE))
    U0 = U1
    if CONVERGE<1e-5:
        break
print("elapsed time: ", time.time()-startt)
U = steadyNS.steadyNS.EmbedU(d,N,NE,B,U1)

#%%
# ALLCONVERGE
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_yscale('log')
ax.plot(ALLCONVERGE)
# quiver
fig = plt.figure(figsize=(maxx//2,maxy//2))
ax = fig.add_subplot(111)
ax.quiver(coordAll[:,0],coordAll[:,1],U[0],U[1],width=0.001)
# velocity x
fig = plt.figure(figsize=(maxx//2+2,maxy//2))
ax = fig.add_subplot(111)
ax.tricontour(coordAll[:,0],coordAll[:,1],U[0],levels=30,linewidths=0.5,colors='k')
cntr = ax.tricontourf(coordAll[:,0],coordAll[:,1],U[0],levels=30,cmap="RdBu_r")
fig.colorbar(cntr,ax=ax)
# velocity y
fig = plt.figure(figsize=(maxx//2+2,maxy//2))
ax = fig.add_subplot(111)
ax.tricontour(coordAll[:,0],coordAll[:,1],U[1],levels=30,linewidths=0.5, colors='k')
cntr = ax.tricontourf(coordAll[:,0],coordAll[:,1],U[1],levels=30,cmap="RdBu_r")
fig.colorbar(cntr,ax=ax)
# velocity norm
fig = plt.figure(figsize=(maxx//2+2,maxy//2))
ax = fig.add_subplot(111)
ax.tricontour(coordAll[:,0],coordAll[:,1],np.sqrt(U[0]**2+U[1]**2),levels=30,linewidths=0.5, colors='k')
cntr = ax.tricontourf(coordAll[:,0],coordAll[:,1],np.sqrt(U[0]**2+U[1]**2),levels=30,cmap="RdBu_r")
fig.colorbar(cntr,ax=ax)
ax.plot(coordAll[B==1,0],coordAll[B==1,1],'ko', ms=5)
ax.plot(coordAll[B==2,0],coordAll[B==2,1],'k>', ms=5)
ax.plot(coordAll[B==3,0],coordAll[B==3,1],'kx', ms=5)
ax.plot(coordAll[B==4,0],coordAll[B==4,1],'kD', ms=2)
# pressure
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
PSelect = np.ndarray(N,dtype=bool)
PSelect[:] = False
# PSelect = np.abs(P)<0.25
for cylinder in Cylinders:
    tmp = np.sqrt((coord[:,0]-cylinder['x'])**2+(coord[:,1]-cylinder['y'])**2)
    tmp = np.abs(tmp-cylinder['d']/2)
    tmp = tmp<1
    PSelect = PSelect | tmp
ax.tricontour(coord[PSelect,0],coord[PSelect,1],P[PSelect],levels=14,linewidths=0.5,colors='k')
cntr = ax.tricontourf(coord[PSelect,0],coord[PSelect,1],P[PSelect],levels=14,cmap="RdBu_r")
fig.colorbar(cntr,ax=ax)

#%%
if len(sys.argv)<=1:
    gmsh.fltk.run()

gmsh.finalize()

#%%

