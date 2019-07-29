import gmsh
import math
import numpy as np
from .mesh_extension import switchEdgeNode,updateEdgeTags

model = gmsh.model
factory = model.geo

def P1Nodes(d, PhysicalWholeDomain, PhysicalInlet, PhysicalOutlet, PhysicalFixWall, PhysicalHoleBoundary):
    # all nodes
    nodeTags,coord = model.mesh.getNodesForPhysicalGroup(dim=d,tag=PhysicalWholeDomain)
    nodeTags -= 1
    nodeTags = nodeTags.astype(np.int32)
    N = len(nodeTags)
    coord = coord.reshape(N,3)
    coord = coord[:,:d]
    coord = coord[nodeTags.argsort()]
    B = np.zeros(N,dtype=np.int32)

    # inlet and outlet nodes
    InletNodeTags,_ = model.mesh.getNodesForPhysicalGroup(dim=d-1,tag=PhysicalInlet)
    InletNodeTags -= 1
    OutletNodeTags,_ = model.mesh.getNodesForPhysicalGroup(dim=d-1,tag=PhysicalOutlet)
    OutletNodeTags -= 1
    B[InletNodeTags] = 1
    B[OutletNodeTags] = 2

    # Fix wall nodes
    FixWallNodeTags,_ = model.mesh.getNodesForPhysicalGroup(dim=d-1,tag=PhysicalFixWall)
    FixWallNodeTags -= 1
    B[FixWallNodeTags] = 3

    # Hole boundary nodes
    NoslipWallNodeTags,_ = model.mesh.getNodesForPhysicalGroup(dim=d-1,tag=PhysicalHoleBoundary)
    NoslipWallNodeTags -= 1
    B[NoslipWallNodeTags] = 4
    return N,coord,B

def P1Elements(d, WholeDomainTag, coord, B):
    _, _, e = model.mesh.getElements(dim=d, tag=WholeDomainTag)
    e = e[0]
    e = np.ascontiguousarray(e.reshape(-1,d+1))-1
    M = len(e)
    e = e.astype(np.int32)
    E = np.empty((M,d+1,d+1),dtype=float)
    E[:,0,:] = 1
    for i in range(d+1):
        E[:,1:,i] = coord[e[:,i]]
    eMeasure = 1/math.gamma(d)*np.abs(np.linalg.det(E))
    E = np.linalg.inv(E)[...,1:]
    E = np.ascontiguousarray(E)
    return M,e,E,eMeasure

def P2Elements(d, B, e, coord):
    EdgeNumPerElem = (d+1)*d//2
    N = len(B)
    M = len(e)
    Edge = np.empty((EdgeNumPerElem*M,2),dtype=np.int32)
    j = 0
    for j1 in range(1,d+1):
        for j2 in range(j1):
            Edge[j*M:(j+1)*M] = e[:,(j2,j1)]
            j += 1
    assert(j==EdgeNumPerElem)
    Edge = switchEdgeNode(Edge)
    Edge,unique_inverse = \
            np.unique(Edge,return_inverse=True,axis=0)
    unique_inverse = unique_inverse.astype(np.int32)
    NE = len(Edge)
    Bedge = updateEdgeTags(d,Edge,B,coord)
    B = np.concatenate((B,Bedge),axis=0)
    elem2edge = unique_inverse.reshape(EdgeNumPerElem, M).transpose()+N
    e = np.concatenate((e,elem2edge),axis=1)
    return NE,B,e,Edge

def P1Check(coord,B,e,Cylinders,maxx=16):
    M = e.shape[0]
    d = e.shape[1]-1
    N = B.shape[0]
    print("node number:",N, "element number:",M)
    # check inlet, outlet, fix wall, hole boundary nodes
    print(np.all(np.abs(coord[B==1,0])<1e-10))
    print(np.all(np.abs(coord[B==2,0]-maxx)<1e-10))
    BC = np.ndarray(N,dtype=bool)
    BC[:] = False
    for cylinder in Cylinders:
        if d==2:
            BCtmp = (coord[:,0]-cylinder['x'])**2+(coord[:,1]-cylinder['y'])**2
        elif d==3:
            BCtmp = (coord[:,0]-cylinder['x'])**2+(coord[:,1]-cylinder['y'])**2+(coord[:,2]-cylinder['z'])**2
        BCtmp -= (cylinder['d']/2)**2
        BCtmp = BCtmp<1e-10
        BC = BC | BCtmp
    print(np.linalg.norm(coord[BC]-coord[B==4]))

def SetHoleTags(coord,B,e,Holes):
    """
    set P1 node tags on different holes,
    this function should be called before P2Elements
    """
    M = e.shape[0]
    d = e.shape[1]-1
    N = B.shape[0]
    idx = np.where(B==4)[0]
    idxForEachHole = []
    for hole in Holes:
        if d==2:
            BCtmp = (coord[idx,0]-hole['x'])**2+(coord[idx,1]-hole['y'])**2
        elif d==3:
            BCtmp = (coord[idx,0]-hole['x'])**2+(coord[idx,1]-hole['y'])**2+(coord[idx,2]-hole['z'])**2
        BCtmp -= (hole['d']/2)**2
        idxForEachHole.append(idx[BCtmp<1e-10])
    assert(sum(len(x) for x in idxForEachHole)==len(idx))
    for k in range(len(idxForEachHole)):
        B[idxForEachHole[k]] = 4+k
    return B

