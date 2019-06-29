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
d = 3;
maxx = 20;
maxy = 8;
maxz = 8;
lcar1 = 0.3;
lcar2 = 0.2;
if len(sys.argv[1:])>=1:
    caseName = sys.argv[1]
else:
    caseName = "spheres"
if len(sys.argv[2:])>=1:
    lcar1 *= float(sys.argv[2])
    lcar2 *= float(sys.argv[2])
argv = sys.argv[3:]
print("characteristic length scale of")
print("box boundary "+str(lcar1))
print("sphere boundary "+str(lcar2))

model = gmsh.model
factory = model.geo

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 1)

model.add("sphere")

#%%
# read spheres configuration 
# if no external configure is available, x=6,y=4,z=4,d=1 is used by default
Spheres = []
if len(argv)<3:
    # Spheres.append(dict(x=6,y=4,z=4,d=1))
    Spheres.append(dict(x=5.5,y=3.5,z=4,d=1))
    Spheres.append(dict(x=6.5,y=4.5,z=4,d=1))
else:
    k = 0
    while len(argv)-k>=4:
        Spheres.append(dict(x=float(argv[k]),y=float(argv[k+1]),z=float(argv[k+2]),d=float(argv[k+3])))
        k += 4

print(Spheres)

#%% construct box
factory.addPoint(   0,   0,   0,lcar1,1); factory.addPoint(maxx,   0,   0,lcar1,2)
factory.addPoint(maxx,maxy,   0,lcar1,3); factory.addPoint(   0,maxy,   0,lcar1,4)
factory.addPoint(   0,   0,maxz,lcar1,5); factory.addPoint(maxx,   0,maxz,lcar1,6)
factory.addPoint(maxx,maxy,maxz,lcar1,7); factory.addPoint(   0,maxy,maxz,lcar1,8)

factory.addLine(  1,  2,  1); factory.addLine(  2,  3,  2)
factory.addLine(  3,  4,  3); factory.addLine(  4,  1,  4)
factory.addLine(  5,  6,  5); factory.addLine(  6,  7,  6)
factory.addLine(  7,  8,  7); factory.addLine(  8,  5,  8)
factory.addLine(  1,  5,  9); factory.addLine(  2,  6, 10)
factory.addLine(  3,  7, 11); factory.addLine(  4,  8, 12)

factory.addCurveLoop([ -1, -2, -3, -4], 13); factory.addCurveLoop([  5,  6,  7,  8], 14)
factory.addCurveLoop([  1, 10, -5, -9], 15); factory.addCurveLoop([  2, 11, -6,-10], 16)
factory.addCurveLoop([  3, 12, -7,-11], 17); factory.addCurveLoop([  4,  9, -8,-12], 18)

# fixwall
factory.addPlaneSurface([13],1); factory.addPlaneSurface([14],2)
factory.addPlaneSurface([15],3); factory.addPlaneSurface([17],5)
factory.addPlaneSurface([16],4) # outlet
factory.addPlaneSurface([18],6) # inlet

shells = []

BoxSL = factory.addSurfaceLoop([1,2,3,4,5,6])
shells.append(BoxSL)
factory.synchronize()

#%% add box physical groups
PhysicalInlet = 1
PhysicalOutlet = 2
PhysicalFixWall = 3
model.addPhysicalGroup(dim=2, tags=[6,], tag=PhysicalInlet)
model.setPhysicalName(dim=2, tag=PhysicalInlet, name='PhysicalInlet')
model.addPhysicalGroup(dim=2, tags=[4,], tag=PhysicalOutlet)
model.setPhysicalName(dim=2, tag=PhysicalOutlet, name='PhysicalOutlet')
model.addPhysicalGroup(dim=2, tags=[1,2,3,5], tag=PhysicalFixWall)
model.setPhysicalName(dim=2, tag=PhysicalFixWall, name='PhysicalFixWall')

#%%
def cheeseHole(x, y, z, d, lc, shells):
    r = d/2
    p1 = factory.addPoint(x,  y,  z,   lc); p2 = factory.addPoint(x+r,y,  z,   lc)
    p3 = factory.addPoint(x,  y+r,z,   lc); p4 = factory.addPoint(x,  y,  z+r, lc)
    p5 = factory.addPoint(x-r,y,  z,   lc); p6 = factory.addPoint(x,  y-r,z,   lc)
    p7 = factory.addPoint(x,  y,  z-r, lc)
    PointTag = [p1,p2,p3,p4,p5,p6,p7]

    c1 = factory.addCircleArc(p2,p1,p7); c2 = factory.addCircleArc(p7,p1,p5)
    c3 = factory.addCircleArc(p5,p1,p4); c4 = factory.addCircleArc(p4,p1,p2)
    c5 = factory.addCircleArc(p2,p1,p3); c6 = factory.addCircleArc(p3,p1,p5)
    c7 = factory.addCircleArc(p5,p1,p6); c8 = factory.addCircleArc(p6,p1,p2)
    c9 = factory.addCircleArc(p7,p1,p3); c10 = factory.addCircleArc(p3,p1,p4)
    c11 = factory.addCircleArc(p4,p1,p6); c12 = factory.addCircleArc(p6,p1,p7)
    CircleArcTag = [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12]

    l1 = factory.addCurveLoop([c5,c10,c4]); l2 = factory.addCurveLoop([c9,-c5,c1])
    l3 = factory.addCurveLoop([c12,-c8,-c1]); l4 = factory.addCurveLoop([c8,-c4,c11])
    l5 = factory.addCurveLoop([-c10,c6,c3]); l6 = factory.addCurveLoop([-c11,-c3,c7])
    l7 = factory.addCurveLoop([-c2,-c7,-c12]); l8 = factory.addCurveLoop([-c6,-c9,c2])
    CurveLoopTag = [l1,l2,l3,l4,l5,l6,l7,l8]

    s1 = factory.addSurfaceFilling([l1]); s2 = factory.addSurfaceFilling([l2])
    s3 = factory.addSurfaceFilling([l3]); s4 = factory.addSurfaceFilling([l4])
    s5 = factory.addSurfaceFilling([l5]); s6 = factory.addSurfaceFilling([l6])
    s7 = factory.addSurfaceFilling([l7]); s8 = factory.addSurfaceFilling([l8])
    SurfaceFillingTag = [s1,s2,s3,s4,s5,s6,s7,s8]

    SurfaceLoopTag = factory.addSurfaceLoop([s1, s2, s3, s4, s5, s6, s7, s8])
    # v = factory.addVolume([sl])
    shells.append(SurfaceLoopTag)
    return PointTag, CircleArcTag, CurveLoopTag, \
            SurfaceFillingTag, SurfaceLoopTag

PointTags = []; CircleArcTags = []
CurveLoopTags = []; SurfaceFillingTags = []
SurfaceLoopTags = []
for sphere in Spheres:
    PointTag, CircleArcTag, CurveLoopTag, \
            SurfaceFillingTag, SurfaceLoopTag = cheeseHole(
                    sphere['x'], sphere['y'], sphere['z'],
                    sphere['d'], lcar2, shells)
    PointTags.append(PointTag); CircleArcTags.append(CircleArcTag)
    CurveLoopTags.append(CurveLoopTag)
    SurfaceFillingTags.append(SurfaceFillingTag)
    SurfaceLoopTags.append(SurfaceLoopTag)
PhysicalHoleBoundary = 4
SurfaceFillingTagsFlat = []
for SurfaceFillingTag in SurfaceFillingTags:
    SurfaceFillingTagsFlat += SurfaceFillingTag
model.addPhysicalGroup(dim=2, tags=SurfaceFillingTagsFlat, tag=PhysicalHoleBoundary)
model.setPhysicalName(dim=2, tag=PhysicalHoleBoundary, name='PhysicalHoleBoundary')
factory.synchronize()

#%% construct computation domain
ComputationDomainTag = 1000
PhysicalWholeDomain = 1000
factory.addVolume(shells, tag=ComputationDomainTag)
model.addPhysicalGroup(dim=3, tags=[ComputationDomainTag,], tag=PhysicalWholeDomain)
model.setPhysicalName(dim=3, tag=PhysicalWholeDomain, name='PhysicalWholeDomain')

#%%
model.mesh.generate(3)

#%% set coordinates and node boundaries
N,coord,B = steadyNS.mesh.P1Nodes(d, 
        PhysicalWholeDomain, PhysicalInlet, PhysicalOutlet, PhysicalFixWall, 
        PhysicalHoleBoundary)
B[B==PhysicalOutlet] = 0

#%% set elements
M,e,E,eMeasure = steadyNS.mesh.P1Elements(d, ComputationDomainTag, coord, B)

steadyNS.mesh.P1Check(coord,B,e,Spheres,maxx)

NE,B,e,Edge = steadyNS.mesh.P2Elements(d, B, e, coord)
coordEdge = (coord[Edge[:,0]]+coord[Edge[:,1]])/2
coordAll = np.concatenate([coord,coordEdge],axis=0)
coordEle = np.zeros((M,3))
for l in range(d+1):
    coordEle += coord[e[:,l]]
coordEle /= d+1
assert(B.shape[0]==NE+N)
for sphere in Spheres:
    tmp = (coordEle[:,0]-sphere['x'])**2+(coordEle[:,1]-sphere['y'])**2\
            +(coordEle[:,2]-sphere['z'])**2
    tmp = np.sqrt(tmp)-sphere['d']/2
    assert(np.all(tmp>1e-10))
print("edge number: ", NE)
print("e.shape: ", e.shape)
DN = (B==0).sum()
print("number of free nodes/edges for velocity: ", DN)

#%% set global stiff matrix for poisson equation
C_NUM = steadyNS.poisson.Poisson_countStiffMatData(d,M,N,NE,B,e)
print("non-zero number of C_OO=",C_NUM)
C = steadyNS.poisson.Poisson_StiffMat(C_NUM,d,nu,M,N,NE,B,e,E,eMeasure)
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
ml = smoothed_aggregation_solver(C,symmetry='hermitian',strength='symmetric')
MM = ml.aspreconditioner(cycle='V')
U = steadyNS.poisson.ReturnU(N,NE,B)
URHIAdd_poisson = C_full@U
URHIAdd_poisson = np.ascontiguousarray(URHIAdd_poisson[B==0])
b = 0.001-URHIAdd_poisson
k = 0
def callback(xk):
    global k
    k += 1
    return print("iter: ", k, "ResNorm: ", np.linalg.norm(C@xk-b))
U,info = sp.sparse.linalg.cg(C,b,tol=1e-10,M=MM,callback=callback)
U = steadyNS.poisson.EmbedU(N,NE,B,U)

#%% show poisson solution
# fig = plt.figure(figsize=(maxx//2+2,maxy//2))
# ax = fig.add_subplot(111)
# ax.tricontour(coordAll[:,0],coordAll[:,1],U,levels=30,linewidths=0.5,colors='k')
# cntr = ax.tricontourf(coordAll[:,0],coordAll[:,1],U,levels=30,cmap="RdBu_r")
# fig.colorbar(cntr,ax=ax)

#%% set global stiff matrix
C0 = steadyNS.steadyNS.StiffMat(d,M,N,NE,B,e,E,eMeasure)
for l in range(d):
    print("C0[",l,"] shape=",C0[l].shape)
    print("C0[",l,"] nnz=",C0[l].nnz)
U = steadyNS.steadyNS.ReturnU(d,N,NE,B)
URHIAdd = np.zeros_like(U)
for l in range(d):
    URHIAdd[l] = C_full@U[l]
URHIAdd = np.ascontiguousarray(URHIAdd[:,B==0])
PRHIAdd = np.zeros(M-1)
for l in range(d):
    PRHIAdd -= C0[l][:-1]@U[l]
for l in range(d):
    C0[l] = C0[l][:-1,B==0]

#%% method 1: solve Schur complement
k = 0
# solveC = sp.sparse.linalg.splu(C).solve
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
    print("solve PRHI: done!")
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
        if k%10==0:
            print("minres IterNum: ", k)
    Ptmp,info = sp.sparse.linalg.minres(BAB, Prhi, tol=1e-7, maxiter=200, callback=callback)
    print("IterNum for solving pressure: ", k, " Precision: ", np.linalg.norm(BAB@Ptmp-Prhi))
    U = np.zeros_like(U0)
    for l in range(d):
        U[l] = solveC(Urhi[l]-C0[l].transpose()@Ptmp)
    return U,Ptmp
U0 = np.zeros((d,DN))
startt = time.time()
for i in range(3):
    U1,P1 = STOKESITE(U0)
    print("Stokes Iter Convergence: ", np.linalg.norm(U0-U1))
    U0 = U1
print("elapsed time: ", time.time()-startt)
U = steadyNS.steadyNS.EmbedU(d,N,NE,B,U0)
P = np.concatenate([P1,np.zeros(1)])

#%% method 2: solve stokes equation directly
BigC = sp.sparse.bmat([[C,None,None],[None,C,None],[None,None,C]])
BigC0 = sp.sparse.bmat([[C0[0],C0[1],C0[2]],])
BigStokes = sp.sparse.bmat([[BigC,BigC0.transpose().tocsr()],[BigC0,None]],format='csr')
def BigStokesmlOP(x):
    y = x.copy()
    for l in range(d):
        y[l*DN:(l+1)*DN] = MM@x[l*DN:(l+1)*DN]
    return y
BigStokesMM = sp.sparse.linalg.LinearOperator(shape=BigStokes.shape, matvec=BigStokesmlOP,rmatvec=BigStokesmlOP)
# BigStokesml = smoothed_aggregation_solver(BigStokes, symmetry='hermitian',strength='symmetric')
# BigStokesMM = BigStokesml.aspreconditioner(cycle='V')
# solveBigStokes = sp.sparse.linalg.splu(BigStokes).solve
def STOKESITE(UP0):
    global k
    U0 = UP0[:d*DN].reshape(d,DN)
    UPrhi = np.zeros_like(UP0)
    UPrhi[:d*DN] = steadyNS.steadyNS.RHI(U0,d,M,N,NE,B,e,E,eMeasure).reshape(-1)
    UPrhi[:d*DN] -= URHIAdd.reshape(-1)
    k = 0
    def callback(xk):
        global k
        if k%200==0:
            print("minres IterNum: ", k, ' ResNorm: ', np.linalg.norm(BigStokes@xk-UPrhi))
        k += 1
    UP,info = sp.sparse.linalg.minres(BigStokes,UPrhi,tol=1e-6,callback=callback,M=BigStokesMM)
    print("IterNum for solving BigStokes: ", k)
    # UP = solveBigStokes(UPrhi)
    print("Solver Precision: ", np.linalg.norm(BigStokes@UP-UPrhi))
    return UP
UP0 = np.zeros(d*DN+M-1)
startt = time.time()
for i in range(3):
    UP1 = STOKESITE(UP0)
    print("Stokes Iter Convergence: ", np.linalg.norm(UP0-UP1))
    UP0 = UP1
print("elapsed time: ", time.time()-startt)
U = UP0[:d*DN].reshape(d,DN)
U = steadyNS.steadyNS.EmbedU(d,N,NE,B,U)
P = np.concatenate([UP0[d*DN:],np.zeros(1)])

#%%
if len(sys.argv)<=1:
    gmsh.fltk.run()
gmsh.finalize()

#%%

