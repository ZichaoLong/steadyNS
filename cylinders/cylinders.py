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
maxx = 20;
maxy = 8;
lcar1 = 0.8;
lcar2 = 0.4;
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
    Cylinders.append(dict(x=5,y=4,d=1))
    Cylinders.append(dict(x=8,y=4,d=1))
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

#%% set elements
M,e,E,eMeasure = steadyNS.mesh.P1Elements(d, WholeDomainTag, coord, B)

steadyNS.mesh.P1Check(coord,B,e,Cylinders,maxx)

NE,B,e = steadyNS.mesh.P2Elements(d, B, e, coord)
assert(B.shape[0]==NE+N)
print("edge number: ", NE)
print("e.shape: ", e.shape)
DN = (B==0).sum()
print("number of free nodes/edges for velocity: ", DN)

#%% set global stiff matrix for poisson equation
C_NUM = steadyNS.poisson.Poisson_countStiffMatData(d,M,N,NE,B,e)
print("non-zero number of C_OO=",C_NUM)
C = steadyNS.poisson.Poisson_StiffMat(C_NUM,d,nu,M,N,NE,B,e,E,eMeasure)
print("C shape=",C.shape)
print("C nnz=",C.nnz)
U = steadyNS.poisson.ReturnU(d,N,NE,B)
URHIAdd = np.zeros_like(U)
for l in range(d):
    URHIAdd[l] = C@U[l]
URHIAdd = np.ascontiguousarray(URHIAdd[:,B==0])
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
b = -URHIAdd[0]
k = 0;
def callback(xk):
    global k
    k += 1
    return print("iter: ", k, "ResNorm: ", np.linalg.norm(C@xk-b))
U,info = sp.sparse.linalg.cg(C,b,tol=1e-10,M=MM,callback=callback)

#%% set global stiff matrix
C0 = steadyNS.steadyNS.StiffMat(d,M,N,NE,B,e,E,eMeasure)
for l in range(d):
    print("C0[",l,"] shape=",C0[l].shape)
    print("C0[",l,"] nnz=",C0[l].nnz)
U = steadyNS.poisson.ReturnU(d,N,NE,B)
PRHIAdd = np.zeros(M-1)
for l in range(d):
    PRHIAdd -= C0[l][:-1]@U[l]
for l in range(d):
    C0[l] = C0[l][:-1,B==0]

#%% method 1: solve Schur complement
k = 0
def STOKESITE(U0):
    global k
    Urhi = steadyNS.steadyNS.RHI(U0,d,M,N,NE,B,e,E,eMeasure)
    Urhi -= URHIAdd
    Prhi = np.zeros(M-1)
    def solveC(x):
        tmp,info = sp.sparse.linalg.cg(C,x,tol=1e-12,M=MM)
        return tmp
    for l in range(d):
        Prhi += C0[l]@solveC(Urhi[l])
    def BABop(x):
        y = np.zeros_like(x)
        for l in range(d):
            y += C0[l]@(solveC(C0[l].transpose()@x))
        return y
    BAB = sp.sparse.linalg.LinearOperator((M-1,M-1), matvec=BABop, rmatvec=BABop)
    k = 0
    def callback(xk):
        global k
        k += 1
    P,info = sp.sparse.linalg.minres(BAB, Prhi, tol=1e-12, maxiter=10, callback=callback)
    print("IterNum for solving pressure: ", k, " Precision: ", np.linalg.norm(BAB@P-Prhi))
    U = np.zeros_like(U0)
    for l in range(d):
        U[l] = solveC(Urhi[l]-C0[l].transpose()@P)
    return U
U0 = np.zeros((d,DN))
startt = time.time()
for i in range(20):
    U1 = STOKESITE(U0)
    print("Stokes Iter Convergence: ", np.linalg.norm(U0-U1))
    U0 = U1
print("elapsed time: ", time.time()-startt)
U = steadyNS.poisson.EmbedU(d,N,NE,B,U0)

#%% method 2: solve stokes equation directly
BigC = sp.sparse.bmat([[C,None],[None,C]])
BigC0 = sp.sparse.bmat([[C0[0],C0[1]],])
BigStokes = sp.sparse.bmat([[BigC,BigC0.transpose().tocsr()],[BigC0,None]],format='csr')
BigStokesml = smoothed_aggregation_solver(BigStokes, symmetry='hermitian',strength='symmetric')
BigStokesMM = BigStokesml.aspreconditioner(cycle='V')
def STOKESITE(UP0):
    global k
    U0 = UP0[:d*DN].reshape(d,DN)
    UPrhi = np.zeros_like(UP0)
    UPrhi[:d*DN] = steadyNS.steadyNS.RHI(U0,d,M,N,NE,B,e,E,eMeasure).reshape(-1)
    UPrhi[:d*DN] -= URHIAdd.reshape(-1)
    k = 0
    def callback(xk):
        global k
        k += 1
    UP,info  = sp.sparse.linalg.minres(BigStokes,UPrhi,tol=1e-12,callback=callback)
    print("IterNum for solving BigStokes: ", k)
    print("Solver Precision: ", np.linalg.norm(BigStokes@UP-UPrhi))
    return UP
UP0 = np.zeros(d*DN+M-1)
startt = time.time()
for i in range(20):
    UP1 = STOKESITE(UP0)
    print("Stokes Iter Convergence: ", np.linalg.norm(UP0-UP1))
    UP0 = UP1
print("elapsed time: ", time.time()-startt)
U = UP0[:d*DN].reshape(d,DN)
U = steadyNS.poisson.EmbedU(d,N,NE,B,U)

#%%
fig = plt.figure(figsize=(maxx//2,maxy//2))
ax = fig.add_subplot(111)
ax.quiver(coord[:,0],coord[:,1],U[0,:N]+1,U[1,:N],width=0.001)

#%%
if len(sys.argv)<=1:
    gmsh.fltk.run()

gmsh.finalize()

#%%

