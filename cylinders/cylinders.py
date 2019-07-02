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

nu = 2
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
    Cylinders.append(dict(x=6,y=4,d=1))
    # Cylinders.append(dict(x=5.5,y=3.5,d=1))
    # Cylinders.append(dict(x=6.5,y=4.5,d=1))
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
C_NUM = steadyNS.poisson.Poisson_countStiffMatData(d,M,N,NE,B,e)
print("non-zero number of C_OO=",C_NUM)
C = steadyNS.poisson.Poisson_StiffMat(C_NUM,d,M,N,NE,B,e,E,eMeasure)
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
ml = smoothed_aggregation_solver(C, symmetry='hermitian',strength='symmetric')
MM = ml.aspreconditioner(cycle='V')
U = steadyNS.poisson.ReturnU(N,NE,B)
URHIAdd_poisson = C_full@U
URHIAdd_poisson = np.ascontiguousarray(URHIAdd_poisson[B==0])
b = 0.01-URHIAdd_poisson
k = 0
def callback(xk):
    global k
    k += 1
    return print("iter: ", k, "ResNorm: ", np.linalg.norm(C@xk-b))
U,info = sp.sparse.linalg.cg(C,b,tol=1e-10,M=MM,callback=callback)
U = steadyNS.poisson.EmbedU(N,NE,B,U)

#%% show poisson solution
fig = plt.figure(figsize=(maxx//2+2,maxy//2))
ax = fig.add_subplot(111)
ax.tricontour(coordAll[:,0],coordAll[:,1],U,levels=30,linewidths=0.5,colors='k')
cntr = ax.tricontourf(coordAll[:,0],coordAll[:,1],U,levels=30,cmap="RdBu_r")
fig.colorbar(cntr,ax=ax)

#%% set pressure part of global stiff matrix
C0 = steadyNS.steadyNS.StiffMat(d,M,N,NE,B,e,E,eMeasure)
for l in range(d):
    print("C0[",l,"] shape=",C0[l].shape)
    print("C0[",l,"] nnz=",C0[l].nnz)
U = steadyNS.steadyNS.ReturnU(d,N,NE,B)
URHIAdd = np.zeros_like(U)
for l in range(d):
    URHIAdd[l] = C_full@U[l]
URHIAdd = np.ascontiguousarray(URHIAdd[:,B==0])
PRHIAdd = np.zeros(M)
for l in range(d):
    PRHIAdd += C0[l]@U[l]
C0_full = list(x for x in C0)
for l in range(d):
    C0[l] = C0[l][:,B==0]

#%% solve stokes equation directly
UP0 = np.zeros(d*DN+M)
BigC = nu*sp.sparse.bmat([[C,None],[None,C]])
BigC0 = sp.sparse.bmat([[C0[0],C0[1]],])
BigStokes = sp.sparse.bmat([[BigC,BigC0.transpose().tocsr()],[BigC0,None]],format='csr')
BigStokesml = smoothed_aggregation_solver(BigStokes,symmetry='hermitian',strength='symmetric')
BigStokesMM = BigStokesml.aspreconditioner(cycle='V')
solveBigStokes = sp.sparse.linalg.splu(BigStokes).solve
def STOKESITE(UP0):
    global k
    U0 = UP0[:d*DN].reshape(d,DN)
    U0 = steadyNS.steadyNS.EmbedU(d,N,NE,B,U0)
    ugu,UplusGU,UGUplus = steadyNS.steadyNS.UGU(U0,d,M,N,NE,B,e,E,eMeasure)
    UPrhi = np.zeros_like(UP0)
    UPrhi[:d*DN] = -ugu[:,B==0].reshape(-1)
    print(nu)
    UPrhi[:d*DN] -= nu*URHIAdd.reshape(-1)
    UPrhi[d*DN:] -= PRHIAdd
    UP = solveBigStokes(UPrhi)
    print("Solver Precision: ", np.linalg.norm(BigStokes@UP-UPrhi))
    return UP
startt = time.time()
for i in range(10):
    UP1 = STOKESITE(UP0)
    print("Stokes Iter Convergence: ", np.linalg.norm(UP0-UP1))
    UP0 = UP1
print("elapsed time: ", time.time()-startt)
U = UP0[:d*DN].reshape(d,DN)
U = steadyNS.steadyNS.EmbedU(d,N,NE,B,U)
P = UP0[d*DN:]
# P = np.concatenate([UP0[d*DN:],np.zeros(1)])
del solveBigStokes
del BigStokes

#%% newton iteration: initial value is from stokes iteration
# UP0 = np.zeros(d*DN+M-1)
# nu = 0.1 # change viscosity here
Bd = np.concatenate([B,]*d,axis=0)
def NEWTONITE(UP0):
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
    BigC = nu*sp.sparse.bmat([[C,None],[None,C]])+UplusGU+UGUplus
    BigC0 = sp.sparse.bmat([[C0[0],C0[1]],])
    BigNewton = sp.sparse.bmat([[BigC,BigC0.transpose().tocsr()],[BigC0,None]],format='csr')
    UP = sp.sparse.linalg.spsolve(BigNewton, UPrhi)
    print("Newton Precision: ", np.linalg.norm(BigNewton@UP-UPrhi))
    return UP
startt = time.time()
for i in range(5):
    UP1 = NEWTONITE(UP0)
    print("Newton Iter Convergence: ", np.linalg.norm(UP0-UP1))
    UP0 = UP1
print("elapsed time: ", time.time()-startt)
U = UP0[:d*DN].reshape(d,DN)
U = steadyNS.steadyNS.EmbedU(d,N,NE,B,U)
P = UP0[d*DN:]
# P = np.concatenate([UP0[d*DN:],np.zeros(1)])

#%%
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
PSelect = np.ndarray(M,dtype=bool)
PSelect[:] = False
# PSelect = np.abs(P)<0.25
for cylinder in Cylinders:
    tmp = np.sqrt((coordEle[:,0]-cylinder['x'])**2+(coordEle[:,1]-cylinder['y'])**2)
    tmp = np.abs(tmp-cylinder['d']/2)
    tmp = tmp<1
    PSelect = PSelect | tmp
ax.tricontour(coordEle[PSelect,0],coordEle[PSelect,1],P[PSelect],levels=14,linewidths=0.5,colors='k')
cntr = ax.tricontourf(coordEle[PSelect,0],coordEle[PSelect,1],P[PSelect],levels=14,cmap="RdBu_r")
fig.colorbar(cntr,ax=ax)

#%%
if len(sys.argv)<=1:
    gmsh.fltk.run()

gmsh.finalize()

#%%

