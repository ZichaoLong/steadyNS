"""
python cylinders.py clscale caseName x1 y1 r1 x2 y2 r2 ...
"""
#%%
import gmsh
import math
import numpy as np
import sys

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
model.mesh.setPeriodic(dim=1, tags=[3,], tagsSource=[1,], affineTransformation=[1,0,0,0, 0,1,0,maxy, 0,0,1,0, 0,0,0,0]);


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
PhysicalPlaneSurface = 100
model.addPhysicalGroup(dim=2, tags=[PlaneSurfaceTag], tag=PhysicalPlaneSurface)
model.setPhysicalName(dim=2, tag=PhysicalPlaneSurface, name='PhysicalPlaneSurface')
factory.synchronize()

#%%
model.mesh.generate(2)

# gmsh.write(caseName+'.msh2')

#%% all nodes
nodeTags,coord = model.mesh.getNodesForPhysicalGroup(dim=2,tag=PhysicalPlaneSurface)
nodeTags -= 1
N = len(nodeTags)
coord = coord.reshape(N,3);
coord = coord[:,:d]
argsortNodeTags = nodeTags.argsort()
coord = coord[argsortNodeTags]
B = np.zeros(N,dtype=int)

#%% inlet and outlet nodes
InletNodeTags,_ = model.mesh.getNodesForPhysicalGroup(dim=1,tag=PhysicalInlet)
InletNodeTags -= 1
OutletNodeTags,_ = model.mesh.getNodesForPhysicalGroup(dim=1,tag=PhysicalOutlet)
OutletNodeTags -= 1
B[InletNodeTags] = 1
B[OutletNodeTags] = 2

#%% noslip wall boundary nodes
NoslipWallNodeTags,_ = model.mesh.getNodesForPhysicalGroup(dim=1,tag=PhysicalCylinderBoundary)
NoslipWallNodeTags -= 1
B[NoslipWallNodeTags] = 3

#%% periodic nodes pair
P = np.zeros(N,dtype=int)-1
tagMaster, nodesPair, affineTransform = model.mesh.getPeriodicNodes(1,3)
nodesPair = np.array(nodesPair)
nodesPair -= 1
nodesPair = nodesPair[B[nodesPair[:,0]]==0]
P[nodesPair[:,0]] = nodesPair[:,1]
for i in range(N):
    if P[i]!=-1:
        if P[P[i]]!=-1:
            P[i] = P[P[i]]
assert(np.all(P[P[P!=-1]]==-1))
B[P!=-1] = 4
B[P[P!=-1]] = 5

#%% check period nodes
print("period")
print(sum(B==4))
print(coord[B==4])
print("period source")
print(sum(B==5))
print(coord[B==5])
#%% check inlet,outlet,noslipwall nodes
print(np.linalg.norm(coord[np.abs(coord[:,0])<1e-10]-coord[B==1]))
print(np.linalg.norm(coord[np.abs(coord[:,0]-maxx)<1e-10]-coord[B==2]))
BC = np.ndarray(N,dtype=bool)
BC[:] = False
for cylinder in Cylinders:
    BCtmp = np.abs((coord[:,0]-cylinder['x'])**2+(coord[:,1]-cylinder['y'])**2-(cylinder['d']/2)**2)<1e-10
    BC = BC | BCtmp
print(np.linalg.norm(coord[BC]-coord[B==3]))
#%% all 2d element
elementTypes, elementTags, nodeTags = model.mesh.getElements(dim=2, tag=PlaneSurfaceTag)
nodeTags = nodeTags[0].reshape(-1,d+1)
M = len(nodeTags)

#%%
if len(sys.argv)<=1:
    gmsh.fltk.run()

gmsh.finalize()

#%%

