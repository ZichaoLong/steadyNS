import gmsh
import math
import numpy as np
from .mesh_extension import reduceP,mergePeriodicNodes,switchEdgeNode,updateEdgeTags

model = gmsh.model
factory = model.geo

def P1Nodes(d, PhysicalWholeDomain, PhysicalInlet, PhysicalOutlet, PhysicalHoleBoundary):
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

    # noslip wall boundary nodes
    NoslipWallNodeTags,_ = model.mesh.getNodesForPhysicalGroup(dim=d-1,tag=PhysicalHoleBoundary)
    NoslipWallNodeTags -= 1
    B[NoslipWallNodeTags] = 3

    # periodic nodes pair
    P = np.zeros(N,dtype=np.int32)-1
    _,nodeTagsSlaver,nodeTagsMaster,_ = model.mesh.getPeriodicNodes(1,3)
    nodesPair = np.stack([nodeTagsSlaver,nodeTagsMaster],axis=-1)
    nodesPair = np.array(nodesPair).astype(np.int32)
    nodesPair -= 1
    # nodesPair = nodesPair[B[nodesPair[:,0]]==0]
    P[nodesPair[:,0]] = nodesPair[:,1]
    P = reduceP(P)
    ## reduceP, implement in python script
    # for i in range(N):
    #     if P[i]!=-1:
    #         if P[P[i]]!=-1:
    #             P[i] = P[P[i]]
    ##
    tmp = P!=-1
    assert(np.all(P[P[tmp]]==-1))
    B[tmp] = 4
    tmp = P[tmp]
    tmp = tmp[B[tmp]==0]
    B[tmp] = -1 # mark all non-boundary source nodes
    return N,coord,B,P

def P1Elements(d, WholeDomainTag, coord, B, P):
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
    e = mergePeriodicNodes(B,P,e)
    return M,e,E,eMeasure

def P2Elements(d, B, e):
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
    NE = len(Edge)
    Bedge = updateEdgeTags(Edge,B)
    B = np.concatenate((B,Bedge),axis=0)
    elem2edge = unique_inverse.reshape(EdgeNumPerElem, M).transpose()+N
    e = np.concatenate((e,elem2edge),axis=1)
    return NE,B,e

def BarycentricCoord(d):
    if d==2:
        w = 1/3
        Lambda = np.ones((d+1,3))/6+np.eye(d+1)/2
    elif d==3:
        alpha = 0.5854101966249685
        beta = 0.138196601125015
        w = 1/4
        Lambda = np.ones((d+1,4))*beta+np.eye(d+1)*(alpha-beta)
    Gamma = w*Lambda
    Theta = Lambda.transpose()@Gamma
    return w,Lambda,Gamma,Theta

def P1Check(coord,B,P,e,Cylinders,maxx=16):
    M = e.shape[0]
    d = e.shape[1]-1
    N = P.size
    assert(np.all(P[e.reshape(-1)]==-1))
    print("node number:",N, "element number:",M)
    print("B==4:", np.argwhere(B==4).reshape(-1))
    print("P!=-1:",np.argwhere(P!=-1).reshape(-1))
    print("B==-1:", np.argwhere(B==-1).reshape(-1))
    print("P[P!=-1]:",P[P!=-1])
    print("period nodes number",sum(B==4))
    print("coord")
    print(coord[B==4].transpose())
    print("period non-boundary source nodes number",sum(B==-1))
    print("coord")
    print(coord[B==-1].transpose())
    # check inlet,outlet,noslipwall nodes
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
    print(np.linalg.norm(coord[BC]-coord[B==3]))
