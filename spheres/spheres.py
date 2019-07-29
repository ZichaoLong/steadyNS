"""
python spheres.py caseName nu dt lcar1 lcar2 x1 y1 r1 x2 y2 r2 ...
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

d = 3;
maxx = 24;
maxy = 8;
maxz = 8;

caseName = "spheres"
nu = 0.1
dt = 0.2
lcar1 = 0.3;
lcar2 = 0.2;
if len(sys.argv[1:])>=1:
    caseName = sys.argv[1]
if len(sys.argv[2:])>=1:
    nu = float(sys.argv[2])
if len(sys.argv[3:])>=1:
    dt = float(sys.argv[3])
if len(sys.argv[4:])>=1:
    lcar1 = float(sys.argv[4])
if len(sys.argv[5:])>=1:
    lcar2 = float(sys.argv[5])
argv = sys.argv[6:]
print("case name: ", caseName)
print("nu: ", nu)
print("dt: ", dt)
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
B = steadyNS.mesh.SetHoleTags(coord,B,e,Spheres)

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
URHIAdd_poisson = steadyNS.utils.CsrMulVec(C_full,U)
URHIAdd_poisson = np.ascontiguousarray(URHIAdd_poisson[B==0])
b = 0.001-URHIAdd_poisson
k = 0
def callback(xk):
    global k
    k += 1
    return print("iter: ", k, "ResNorm: ", np.linalg.norm(steadyNS.utils.CsrMulVec(C,xk)-b))
print("test poisson solver")
U,info = sp.sparse.linalg.cg(C,b,tol=1e-10,M=PoissonMM,callback=callback)
U = steadyNS.poisson.EmbedU(N,NE,B,U)

#%% show poisson solution
# dx = 0.1
# uniformU = steadyNS.utils.InterpP2ToUniformGrid(dx,maxx,maxy,maxz,U,M,N,NE,e,coordAll)
# selectzidx = int(maxz*0.5/dx)
# fig = plt.figure()
# ax = fig.add_subplot(111)
# colorPoisson = ax.imshow(uniformU[...,selectzidx].transpose(),cmap='jet')
# fig.colorbar(colorPoisson,ax=ax)

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
sys.stdout.flush()

#%% set URHIADD, PRHIADD
U = steadyNS.steadyNS.ReturnU(d,N,NE,B)
URHIAdd = np.zeros_like(U)
for l in range(d):
    URHIAdd[l] = steadyNS.utils.CsrMulVec(C_full,U[l])
URHIAdd = np.ascontiguousarray(URHIAdd[:,B==0])
PRHIAdd = np.zeros(N)
for l in range(d):
    PRHIAdd += steadyNS.utils.CsrMulVec(C0_full[l],U[l])

#%% pressure correction method step 0: set linear system
PrintInfo = False
# BigC = nu*sp.sparse.bmat([[C,None],[None,C]])
# solveBigStokes = sp.sparse.linalg.splu(BigStokes).solve

STEP0LinearSystem = (CF/dt+nu*C).tocsr()
STEP0ML = smoothed_aggregation_solver(STEP0LinearSystem,symmetry='hermitian',strength='symmetric')
STEP0MM = STEP0ML.aspreconditioner(cycle='V')
STEP0LinearSystemOPFUNC = lambda x:steadyNS.utils.CsrMulVec(STEP0LinearSystem, x)
STEP0LinearSystemOP = sp.sparse.linalg.LinearOperator(shape=STEP0LinearSystem.shape,
        matvec=STEP0LinearSystemOPFUNC, rmatvec=STEP0LinearSystemOPFUNC)
def STEP0(U0,tol=1e-10):
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
            print("iter: ", k, " ResNorm: ", np.linalg.norm(STEP0LinearSystemOP@xk-b))
        return 
    Utilde = np.zeros((d,DN))
    tmp = 0
    iternum = 0
    for l in range(d):
        k = 0
        b = STEP0rhi[l]
        Utilde[l],iternumtmp = steadyNS.utils.CG(STEP0LinearSystemOP, b, 
                x0=U0[l], tol=tol, maxiter=50, 
                M=STEP0MM, callback=callback)
        tmp +=  np.linalg.norm(STEP0LinearSystemOP@Utilde[l]-STEP0rhi[l])
        iternum += iternumtmp
    print("STEP0 IterNum: ", iternum, " Precision: {:.2e}".format(tmp))
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
def STEP1(Utilde,P0=None,tol=1e-10):
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
            print("iter: ", k, " ResNorm: ", np.linalg.norm(PLinearSystem@xk-Prhi))
        return 
    k = 0
    P0 = (np.zeros_like(Prhi) if P0 is None else P0)
    P,iternum = steadyNS.utils.CG(PLinearSystem,Prhi,x0=P0,tol=tol,maxiter=50,M=PrePLinearSystemMM,callback=callback)
    print("STEP1 IterNum: ", iternum, " Precision: {:.2e}".format(np.linalg.norm(PLinearSystem@P-Prhi)))
    for l in range(d):
        U[l] += dt*CFSolver(steadyNS.utils.CsrMulVec(C0transpose[l],P))
    return U,P

#%% test pressure correction method step0, step1
# PrintInfo = True
# U0 = np.random.randn(d,DN)/10
# startt = time.time()
# Utilde = STEP0(U0)
# print("step0 elapsed time: ", time.time()-startt)
# startt = time.time()
# U1,P = STEP1(Utilde)
# print("step1 elapsed time: ", time.time()-startt)

#%% pressure correction method
U0 = np.zeros((d,DN))
ALLCONVERGE = []
tol = 1e-5
PrintInfo = False
TotalStartTime = time.time()
startt = time.time()
for steps in range(2000):
    Utilde = STEP0(U0,tol=tol)
    if steps==0:
        U1,P = STEP1(Utilde,tol=tol)
    else:
        U1,P = STEP1(Utilde,tol=tol,P0=P)
    CONVERGE = np.abs(U1-U0).max()/dt
    if CONVERGE<3e-4:
        tol = 1e-10
    ALLCONVERGE.append(CONVERGE)
    DIVERGENCENORM = PRHIAdd
    for l in range(d):
        DIVERGENCENORM = DIVERGENCENORM+steadyNS.utils.CsrMulVec(C0[l],U1[l])
    DIVERGENCENORM = np.linalg.norm(DIVERGENCENORM)
    print("ITER: ", steps, " ElapsedTime: {:.2e}".format(time.time()-startt), 
            " div U1: {:.2e}".format(DIVERGENCENORM), " max(|U1-U0|/dt): {:.2e}".format(CONVERGE))
    sys.stdout.flush()
    startt = time.time()
    U0 = U1
    if CONVERGE<1e-4:
        break
    if CONVERGE>1e5:
        break
print("ITER: ", steps, " Total Elapsed Time: {:.2e}".format(time.time()-TotalStartTime))
U = steadyNS.steadyNS.EmbedU(d,N,NE,B,U1)

#%% interp to uniform grid
dx = 0.1
uniformU = []
for l in range(d):
    uniformU.append(steadyNS.utils.InterpP2ToUniformGrid(dx,maxx,maxy,maxz,U[l],M,N,NE,e,coordAll))
uniformP = steadyNS.utils.InterpP1ToUniformGrid(dx,maxx,maxy,maxz,P,M,N,NE,e,coordAll)
selectzidx = int(maxz*0.5/dx)
fig = plt.figure()
ax1 = fig.add_subplot(4,1,1)
coloru = ax1.imshow(uniformU[0][...,selectzidx].transpose(),cmap='jet')
ax2 = fig.add_subplot(4,1,2)
colorv = ax2.imshow(uniformU[1][...,selectzidx].transpose(),cmap='jet')
ax3 = fig.add_subplot(4,1,3)
colorw = ax3.imshow(uniformU[2][...,selectzidx].transpose(),cmap='jet')
ax4 = fig.add_subplot(4,1,4)
colorp = ax4.imshow(uniformP[...,selectzidx].transpose(),cmap='jet')
fig.colorbar(coloru,ax=ax1)
fig.colorbar(colorv,ax=ax2)
fig.colorbar(colorw,ax=ax3)
fig.colorbar(colorp,ax=ax4)
#%%
if len(sys.argv)<=1:
    gmsh.fltk.run()
gmsh.finalize()

#%%

