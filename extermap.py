# reference:
#  T. A. Driscoll and L. N. Trefethen
#   "Schwarz-Christoffel Mapping" (Cambridge University Press)

import numpy as np
from scipy.special import roots_legendre, roots_jacobi
from scipy.optimize import root
import matplotlib.pyplot as plt

class extermap:
    """ map from interior of unit disk into exterior of polygon
    """
    def __init__(self, polygon, n_node=8, method='krylov'):
        """
        polygon = target polygon of SC transformation
        n_node = number of nodes for gaussian quadrature
        method = used for root finding
        """
        if np.any(np.isinf(polygon.vertex)):
            print('infinite vertex is not allowed'); exit()

        p = polygon.copy()
        p.flip()
        p.roll(1)

        w = p.vertex
        b = p.angle
        b[b==-1] = 1
        n = len(w)

        node = np.empty([n+1, n_node])
        weight = np.empty_like(node)

        for k in range(n):
            if np.isfinite(w[k]):
                (node[k], weight[k]) = roots_jacobi(n_node, 0, b[k])

        (node[n], weight[n]) = roots_legendre(n_node)

        self.prevertex = np.empty(n, dtype=np.complex)
        self.vertex = w
        self.angle = b
        self.node = node
        self.weight = weight
        self.map = np.vectorize(self.map)

        y = np.zeros(n-1)
        f = np.empty_like(y)

        def scfun(y):
            z = self.yztran(y)
            C = (w[-1] - w[0])/self.zquad(z[0], z[-1], 0, n-1)

            for k in range(n-3):
                q = self.zquad(z[k], z[k+1], k, k+1)
                f[k] = np.abs(w[k+1] - w[k]) - np.abs(C*q)
 
            r = np.sum(b/z)
            f[n-3] = np.real(r)
            f[n-2] = np.imag(r)

            self.C = C
            return f

        sol = root(scfun, y, method=method, options={'disp': True})
        self.yztran(sol.x)

    def yztran(self,y):
        y = 1 + np.cumsum(np.exp(-np.cumsum(y)))
        t = 2 * np.pi / y[-1]
        self.prevertex[0] = np.exp(t * 1j)
        self.prevertex[1:] = np.exp(t * y * 1j)
        return self.prevertex

    def zprod(self,z,k=-1):
        t = 1 - np.outer(1/self.prevertex, z)
        if k>=0: t[k] /= np.abs(t[k])
        return np.exp(np.dot(self.angle, np.log(t)))/z**2

    def dist(self,z,k):
        d = np.abs(z - self.prevertex)
        if k>=0: d[k] = np.inf
        return min(np.min(d), np.abs(z))

    def zqsum(self,za,zb,k):
        if za==zb: return 0
        h = (zb-za)/2
        t = self.zprod((za+zb)/2 + h*self.node[k], k)
        t = h * np.dot(self.weight[k], t)
        if k>=0: t *= np.abs(h)**self.angle[k]
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
        if np.abs(np.angle(-zb/za)) < 1.e-3:
            zm *= 0.5/np.abs(zm)
        return self.zquad1(za,zm,ka) - self.zquad1(zb,zm,kb)

    def map(self, z, k=-1):
        """ SC transformation of z """
        zk = self.prevertex
        if k<0: k = np.argmin(np.abs(z-zk))
        return self.vertex[k] + self.C * self.zquad(zk[k],z,k)

    def plot(self, r, theta, *arg, **kwarg):
        """
        r = radius of circles in disk
        theta = direction of rays in disk
        arg = arguments passed to plt.plot
        kwarg = keyword arguments passed to plt.plot
        """
        N = 256; EPS = 1.e-3

        if len(r):
            r,th = np.meshgrid(r, np.linspace(0, 2*np.pi, N))
            w = self.map(r * np.exp(th * 1j))
            plt.plot(np.real(w), np.imag(w), *arg, **kwarg)

        if len(theta):
            r,th = np.meshgrid(np.linspace(EPS,1-EPS,N), theta)
            w = self.map(r.T * np.exp(th.T * 1j))
            plt.plot(np.real(w), np.imag(w), *arg, **kwarg)
