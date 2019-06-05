"""
python cylinders.py caseName clscale x1 y1 r1 x2 y2 r2 ...
"""
#%%
import gmsh
import math
import numpy as np
import sys
import steadyNS

nu = 0.1
d = 2;
maxx = 16;
maxy = 4;
lcar1 = 1;
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
    Cylinders.append(dict(x=4,y=2,d=1))
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
model.mesh.setPeriodic(dim=1, tags=[3,], tagsMaster=[1,], affineTransform=[1,0,0,0, 0,1,0,maxy, 0,0,1,0, 0,0,0,0]);


#%% add box physical groups
PhysicalInletNodes = 1
PhysicalOutletNodes = 2
PhysicalPeriodBoundary = 4
PhysicalInlet = 1
PhysicalOutlet = 2
model.addPhysicalGroup(dim=0, tags=[1,4], tag=PhysicalInletNodes)
model.setPhysicalName(dim=0, tag=PhysicalInletNodes, name='PhysicalInletNodes')
model.addPhysicalGroup(dim=0, tags=[2,3], tag=PhysicalOutletNodes)
model.setPhysicalName(dim=0, tag=PhysicalOutletNodes, name='PhysicalOutletNodes')
model.addPhysicalGroup(dim=1, tags=[1,3], tag=PhysicalPeriodBoundary)
model.setPhysicalName(dim=1, tag=PhysicalPeriodBoundary, name='PhysicalPeriodBoundary')
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
PhysicalCylinderBoundary = 3
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
N,coord,B,P = steadyNS.mesh.P1Nodes(d, 
        PhysicalWholeDomain, PhysicalInlet, PhysicalOutlet, PhysicalHoleBoundary)

#%% set elements
M,e,E,eMeasure = steadyNS.mesh.P1Elements(d, WholeDomainTag, coord, B, P)

steadyNS.mesh.P1Check(coord,B,P,e,Cylinders,maxx=16)

#%% set barycentric coordinate
w,Lambda,Gamma,Theta = steadyNS.mesh.BarycentricCoord(d)
print("w",w)
print("Lambda\n",Lambda)
print("Gamma\n", Gamma)
print("Theta\n", Theta)

#%% set global stiff matrix
C_NUM = steadyNS.steadyNS.countStiffMatData(B,P,e)
print("non-zero number of C_OO=",C_NUM)
C = steadyNS.steadyNS.StiffMat(C_NUM,nu,B,P,e,E,eMeasure)
print("C shape=",C.shape)
print("C nnz=",C.nnz)
print("condition number of C=",np.linalg.cond(C.todense()))
C = C.todense()
values,vectors = np.linalg.eig(C)
vectors = np.array(vectors)
zerovectors = vectors[:,np.abs(values.reshape(-1))<1e-10]
print(np.linalg.norm(C[:-1,324:]@zerovectors[324:]))
index = np.ndarray(C.shape[0],dtype=np.bool)
index[:] = True
for i in range(N):
    if (B[i]==1 or B[i]==2 or B[i]==3):
        for l in range(d):
            index[d*i+l] = False
print("condition number of reduceC=",np.linalg.cond(C[index][:,index]))

#%% set stiff matrix for poisson equation
C_NUM = steadyNS.steadyNS.countPoisson(B,P,e)
print("non-zero number of C_OO=",C_NUM)
C = steadyNS.steadyNS.Poisson(C_NUM,nu,B,P,e,E,eMeasure)
print("C shape=",C.shape)
print("C nnz=",C.nnz)
print("condition number of C=",np.linalg.cond(C.todense()))

#%%
if len(sys.argv)<=1:
    gmsh.fltk.run()

gmsh.finalize()

#%%

