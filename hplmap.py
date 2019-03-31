# reference:
#  T. A. Driscoll and L. N. Trefethen
#   "Schwarz-Christoffel Mapping" (Cambridge University Press)

import numpy as np
from scipy.special import roots_legendre, roots_jacobi
from scipy.optimize import root, newton
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

class hplmap:
    """ map from upper half plane into interior of polygon
    """
    def __init__(self, polygon, n_node=8, method='krylov'):
        """
        polygon = target polygon of SC transformation
        n_node = number of nodes for gaussian quadrature
        method = used for root finding
        """
        p = polygon.copy()
        p.roll()

        w = p.vertex
        beta = p.angle
        n = len(w)

        node = np.empty([n+1, n_node])
        weight = np.empty_like(node)

        for k in range(n):
            if np.isinf(w[k]): continue
            (node[k], weight[k]) = roots_jacobi(n_node, 0, -beta[k])

        (node[n], weight[n]) = roots_legendre(n_node)

        self.prevertex = np.empty(n, dtype=np.complex)
        self.prevertex[[0,-1,-2]] = [1,0,np.inf]
        self.vertex = w
        self.angle = beta
        self.node = node
        self.weight = weight
        self.map = np.vectorize(self.map)
        self.invmap = np.vectorize(self.invmap)

        if n==3:
            self.C = (w[0] - w[-1])/self.zquad(0,1,n-1,0)
            return

        k_rat = p.FiniteEdge([0,n-3])

        k_fix = p.InfVertex([1,n-3])

        y = np.zeros(n-3)
        f = np.empty_like(y)

        def scfun(y):
            z = self.yztran(y)
            C = (w[0] - w[-1])/self.zquad(0,1,n-1,0)

            i = 0
            for k in k_rat:
                q = self.zquad(z[k], z[k+1], k, k+1)
                f[i] = np.abs(w[k+1] - w[k]) - np.abs(C*q)
                i += 1

            for k in k_fix:
                zm = (z[k-1] + z[k+1] + (z[k+1] - z[k-1])*1j)/2
                q = self.zquad(z[k-1], zm, k-1) - self.zquad(z[k+1], zm, k+1)
                q = w[k+1] - w[k-1] - C*q
                f[i] = np.real(q)
                f[i+1] = np.imag(q)
                i += 2
 
            self.C = C
            return f

        sol = root(scfun, y, method=method, options={'disp': True})
        self.yztran(sol.x)

    def yztran(self,y):
        self.prevertex[1:-2] = 1 + np.cumsum(np.exp(-np.cumsum(y)))
        return self.prevertex

    def zprod(self,z,k=-1):
        t = z - self.prevertex[:, np.newaxis]
        if k>=0: t[k] /= np.abs(t[k])
        t[np.isinf(self.prevertex)] = 1
        return np.exp(-np.dot(self.angle, np.log(t)))

    def dist(self,z,k):
        d = np.abs(z - self.prevertex)
        if k>=0: d[k] = np.inf
        return np.min(d)

    def zqsum(self,za,zb,k):
        if za==zb: return 0
        h = (zb-za)/2
        t = self.zprod((za+zb)/2 + h*self.node[k], k)
        t = h * np.dot(self.weight[k], t)
        if k>=0: t /= np.abs(h)**self.angle[k]
        return t

    def zquad1(self,za,zb,ka):
        if za==zb: return 0
        q=0
        for _ in range(100):
            R = min(1, 2*self.dist(za,ka)/np.abs(zb-za))
            zaa = za + R*(zb-za)
            q += self.zqsum(za,zaa,ka)
            if R==1: return q
            za = zaa
            ka = -1

        print('zquad1 failed'); exit()

    def zquad(self, za, zb, ka=-1, kb=-1):
        zm = (za+zb)/2
        return self.zquad1(za,zm,ka) - self.zquad1(zb,zm,kb)

    def map(self, z, k=-1):
        """ SC transformation of z """
        zk = self.prevertex
        if k<0:
            d = np.abs(z - zk)
            d[np.isinf(self.vertex)] = np.inf
            k = np.argmin(d)

        return self.vertex[k] + self.C * self.zquad(zk[k],z,k)

    def invmap(self, w):
        """ inverse SC transformation of w """
        k = np.argmax(self.angle)
        zk = self.prevertex[k]
        c = (w - self.vertex[k])/self.C

        s = solve_ivp(lambda t,z: c/self.zprod(z) if z!=zk else 0,
                      [0,1], [zk])

        return newton(lambda z: self.zquad(zk,z,k) - c, s.y[0,-1],
                      lambda z: self.zprod(z))

    def plot(self, X, Y, *arg, **kwarg):
        """
        X,Y = x,y coordinates of mesh in upper half-plane
        arg = arguments passed to plt.plot
        kwarg = keyword arguments passed to plt.plot
        """
        N = 256; EPS = 1.e-4

        if len(X)>2:
            x,y = np.meshgrid(X, np.linspace(EPS, np.max(Y), N))
            w = self.map(x + y*1j)
            plt.plot(np.real(w), np.imag(w), *arg, **kwarg)

        if len(Y)>1:
            x,y = np.meshgrid(np.linspace(np.min(X), np.max(X), N), Y)
            w = self.map(x.T + y.T*1j)
            plt.plot(np.real(w), np.imag(w), *arg, **kwarg)
