import gmsh
import math
import numpy as np
from .steadyNS import reduceP,mergePeriodicNodes

model = gmsh.model
factory = model.geo

def nodesPreprocess(d, PhysicalWholeDomain, PhysicalInlet, PhysicalOutlet, PhysicalHoleBoundary):
    # all nodes
    nodeTags,coord = model.mesh.getNodesForPhysicalGroup(dim=d,tag=PhysicalWholeDomain)
    nodeTags -= 1
    N = len(nodeTags)
    coord = coord.reshape(N,3)
    coord = coord[:,:d]
    coord = coord[nodeTags.argsort()]
    B = np.zeros(N,dtype=int)

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
    P = np.zeros(N,dtype=int)-1
    _, nodesPair, _ = model.mesh.getPeriodicNodes(1,3)
    nodesPair = np.array(nodesPair)
    nodesPair -= 1
    nodesPair = nodesPair[B[nodesPair[:,0]]==0]
    P[nodesPair[:,0]] = nodesPair[:,1]
    P = reduceP(P)
    ## reduceP, implement in python script
    # for i in range(N):
    #     if P[i]!=-1:
    #         if P[P[i]]!=-1:
    #             P[i] = P[P[i]]
    ##
    assert(np.all(P[P[P!=-1]]==-1))
    B[P!=-1] = 4
    B[P[P!=-1]] = 5
    return N,coord,B,P

def nodesCheck(d,N,coord,B,P,Cylinders,maxx=16):
    # check period nodes
    print("node number",N)
    print("periodic nodes ", np.argwhere(B==4).reshape(-1))
    print("source nodes ", np.argwhere(B==5).reshape(-1))
    print("period nodes number",sum(B==4))
    print("coord")
    print(coord[B==4].transpose())
    print("period source nodes number",sum(B==5))
    print("coord")
    print(coord[B==5].transpose())
    # check inlet,outlet,noslipwall nodes
    print(np.linalg.norm(coord[np.abs(coord[:,0])<1e-10]-coord[B==1]))
    print(np.linalg.norm(coord[np.abs(coord[:,0]-maxx)<1e-10]-coord[B==2]))
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

def elementPreprocess(d, coord, B, P, e):
    e = np.ascontiguousarray(e.reshape(-1,d+1))-1
    M = len(e)
    e = e.astype(int)
    E = np.empty((M,d+1,d+1))
    E[:,0,:] = 1
    for i in range(d+1):
        E[:,1:,i] = coord[e[:,i]]
    eMeasure = math.gamma(d)*np.abs(np.linalg.det(E))
    E = np.linalg.inv(E)[...,1:]
    E = np.ascontiguousarray(E)
    e = mergePeriodicNodes(B,P,e)
    return M,e,E,eMeasure
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
