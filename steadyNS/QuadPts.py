import numpy as np

__all__ = ["quadpts","quadpts2","quadpts3"]

def quadpts(d, order):
    """
    See ifem/fem/quadpts.m
    References: 
    * L. Chen. iFEM: an integrated finite element method package in MATLAB. 
       Technical Report, University of California at Irvine, 2009. 
    * David Dunavant. High degree efficient symmetrical Gaussian
       quadrature rules for the triangle. International journal for numerical
       methods in engineering. 21(6):1129--1148, 1985. 
    * John Burkardt. DUNAVANT Quadrature Rules for the Triangle.
       http://people.sc.fsu.edu/~burkardt/m_src/dunavant/dunavant.html
    """
    if d==2:
        W,Lambda = quadpts2(order)
    elif d==3:
        W,Lambda = quadpts3(order)
    W = np.ascontiguousarray(W)
    Lambda = np.ascontiguousarray(Lambda)
    return W,Lambda

def quadpts2(order):
    if order>5:
        order = 5
    if order==1:
        nQuad = 1
        Lambda = np.array([[1/3, 1/3, 1/3]])
        weight = np.array([1],dtype=np.float64)
    elif order==2:
        nQuad = 3
        Lambda = np.array([
            [2/3, 1/6, 1/6],
            [1/6, 2/3, 1/6],
            [1/6, 1/6, 2/3]])
        weight = np.array([1/3, 1/3, 1/3])
    elif order==3:
        nQuad = 4
        Lambda = np.array([
            [1/3, 1/3, 1/3],
            [0.6, 0.2, 0.2],
            [0.2, 0.6, 0.2],
            [0.2, 0.2, 0.6]])
        weight = np.array([-27/48, 25/48, 25/48, 25/48])
    elif order==4:
        nQuad = 6
        Lambda = np.array([
            [0.108103018168070, 0.445948490915965, 0.445948490915965],
            [0.445948490915965, 0.108103018168070, 0.445948490915965],
            [0.445948490915965, 0.445948490915965, 0.108103018168070],
            [0.816847572980459, 0.091576213509771, 0.091576213509771],
            [0.091576213509771, 0.816847572980459, 0.091576213509771],
            [0.091576213509771, 0.091576213509771, 0.816847572980459]])
        weight = np.array([0.223381589678011, 0.223381589678011, 0.223381589678011,
            0.109951743655322, 0.109951743655322, 0.109951743655322])
    elif order==5:
        nQuad = 7
        alpha1 = 0.059715871789770;      beta1 = 0.470142064105115;
        alpha2 = 0.797426985353087;      beta2 = 0.101286507323456;
        Lambda = np.array([
            [   1/3,    1/3,    1/3],
            [alpha1,  beta1,  beta1],
            [ beta1, alpha1,  beta1],
            [ beta1,  beta1, alpha1],
            [alpha2,  beta2,  beta2],
            [ beta2, alpha2,  beta2],
            [ beta2,  beta2, alpha2]])
        weight = np.array([0.225, 0.132394152788506, 0.132394152788506, 
            0.132394152788506, 0.125939180544827, 
            0.125939180544827, 0.125939180544827])
    return weight,Lambda

def quadpts3(order):
    if order>5:
        order = 5
    if order==1:
        nQuad = 1
        Lambda = np.array([[1/4, 1/4, 1/4, 1/4]])
        weight = np.array([1],dtype=np.float64)
    elif order==2:
        nQuad = 4
        alpha = 0.5854101966249685; 
        beta =  0.138196601125015;
        Lambda = np.array([
            [alpha,beta,beta,beta],
            [beta,alpha,beta,beta],
            [beta,beta,alpha,beta],
            [beta,beta,beta,alpha]])
        weight = np.array([1/4, 1/4, 1/4, 1/4])
    elif order==3:
        nQuad = 5
        Lambda = np.array([
            [1/4,1/4,1/4,1/4],
            [1/2,1/6,1/6,1/6],
            [1/6,1/2,1/6,1/6],
            [1/6,1/6,1/2,1/6],
            [1/6,1/6,1/6,1/2]])
        weight = np.array([-4/5, 9/20, 9/20, 9/20, 9/20])
    elif order==4:
        nQuad = 16
        alpha1 = 0.7716429020672371; 
        beta1 =  0.7611903264425430e-1;
        w1 = 0.5037379410012282e-1;
        alpha = 0.4042339134672644;
        beta = 0.7183164526766925e-1;
        gamma = 0.11970052777978019;
        w2 = 0.6654206863329239e-1;
        Lambda = np.array([
            [alpha1,beta1,beta1,beta1],
            [beta1,alpha1,beta1,beta1],
            [beta1,beta1,alpha1,beta1],
            [beta1,beta1,beta1,alpha1],
            [alpha,alpha,beta,gamma],
            [alpha,alpha,gamma,beta],
            [alpha,beta,alpha,gamma],
            [alpha,beta,gamma,alpha],
            [alpha,gamma,beta,alpha],
            [alpha,gamma,alpha,beta],
            [beta,alpha,alpha,gamma],
            [beta,alpha,gamma,alpha],
            [beta,gamma,alpha,alpha],
            [gamma,alpha,alpha,beta],
            [gamma,alpha,beta,alpha],
            [gamma,beta,alpha,alpha]])                  
        weight = np.array([w1, w1, w1, w1,
                  w2, w2, w2, w2, w2, w2,
                  w2, w2, w2, w2, w2, w2])
    elif order==5:
        nQuad = 17
        alpha1 = 0.7316369079576180; 
        beta1 =  0.8945436401412733e-1;
        w1 = 0.6703858372604275e-1;
        alpha = 0.4214394310662522;
        beta = 0.2454003792903000e-1;
        gamma = 0.1325810999384657;
        w2 = 0.4528559236327399e-1;
        Lambda = np.array([
            [1/4, 1/4, 1/4, 1/4],
            [alpha1,beta1,beta1,beta1],
            [beta1,alpha1,beta1,beta1],
            [beta1,beta1,alpha1,beta1],
            [beta1,beta1,beta1,alpha1],
            [alpha,alpha,beta,gamma],
            [alpha,alpha,gamma,beta],
            [alpha,beta,alpha,gamma],
            [alpha,beta,gamma,alpha],
            [alpha,gamma,beta,alpha],
            [alpha,gamma,alpha,beta],
            [beta,alpha,alpha,gamma],
            [beta,alpha,gamma,alpha],
            [beta,gamma,alpha,alpha],
            [gamma,alpha,alpha,beta],
            [gamma,alpha,beta,alpha],
            [gamma,beta,alpha,alpha]])
        weight = np.array([0.1884185567365411,
                  w1, w1, w1, w1,
                  w2, w2, w2, w2, w2, w2,
                  w2, w2, w2, w2, w2, w2]);
    return weight,Lambda

def _test_quadpts(d):
    def f1(x,n):
        return (x**n).sum()
    def f2(x):
        return np.sin(x.sum())
    if d==2:
        node = np.array([[0,0],[1,0],[0,1]],dtype=np.float64)
        elem = [0,1,2]
        measure = 0.5
    elif d==3:
        node = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],dtype=np.float64)
        elem = [0,1,2,3]
        measure = 1/6
    err1 = np.zeros((5,5))
    err2 = np.zeros((5,5))
    for order in range(1,6):
        W,Lambda = quadpts(d,order)
        nQuad = W.size
        for n in range(1,6):
            t1 = 0
            t2 = 0
            for p in range(nQuad):
                coord = np.zeros(d)
                for i in range(d+1):
                    coord += Lambda[p,i]*node[i]
                t1 += W[p]*f1(coord,n)
                t2 += W[p]*f2(coord)
            t1 *= measure
            t2 *= measure
            if d==2:
                err1[order-1,n-1] = abs(t1-2/((n+1)*(n+2)))
                err2[order-1,n-1] = abs(t2-(np.sin(1)-np.cos(1)))
            elif d==3:
                err1[order-1,n-1] = abs(t1-3/((n+1)*(n+2)*(n+3)))
                err2[order-1,n-1] = abs(t2-(np.sin(1)+1/2*np.cos(1)-1))
    print("row: quadrature points order")
    print("column: polynomial order")
    print(err1)
    print("\int_e sin(sum(coord)). row: quadrature points order")
    print(err2[:,0].transpose())

if __name__ == "__main__":
    np.set_printoptions(precision=3)
    print("test 2d")
    _test_quadpts(2)
    print("\ntest 3d")
    _test_quadpts(3)
