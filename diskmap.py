# reference:
#  L. N. Trefethen
#   "Numerical Computation of the Schwarz-Christoffel Transformation"
#   SIAM Journal on Scientific and Statistical Computing 1 (1980) 82
# derived from:
#  SCPACK
#   www.netlib.org/conformal

import numpy as np
from scipy.special import roots_legendre, roots_jacobi
from scipy.optimize import root, newton
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

class diskmap:
    """ map from interior of unit disk into interior of polygon
    """
    def __init__(self, polygon, center=0, n_node=8, method='krylov'):
        """
        polygon = target polygon of SC transformation
        center = point in polygon where disk center is mapped to
        n_node = number of nodes for gaussian quadrature
        method = used for root finding
        """
        p = polygon.copy()
        p.roll()

        w = p.vertex
        beta = p.angle
        n = len(w)

        k_rat = p.FiniteEdge([0,n-1], n-3)
        k_fix = p.InfVertex()
        if len(k_fix)==0: k_fix = [1]

        node = np.empty([n+1, n_node])
        weight = np.empty_like(node)

        for k in range(n):
            if np.isinf(w[k]): continue
            (node[k], weight[k]) = roots_jacobi(n_node, 0, -beta[k])

        (node[n], weight[n]) = roots_legendre(n_node)

        self.prevertex = np.empty(n, dtype=np.complex)
        self.vertex = w
        self.angle = beta
        self.node = node
        self.weight = weight
        self.A = center
        self.map = np.vectorize(self.map)
        self.invmap = np.vectorize(self.invmap)

        y = np.zeros(n-1)
        f = np.empty_like(y)

        def scfun(y):
            z = self.yztran(y)
            C = (center - w[-1])/self.zquad(1, 0, n-1)

            i = 0
            for k in k_fix:
                q = w[k-1] - center + C*self.zquad(z[k-1], 0, k-1)
                f[i] = np.real(q)
                f[i+1] = np.imag(q)
                i += 2
 
            for k in k_rat:
                q = self.zquad(z[k], z[k+1], k, k+1)
                f[i] = np.abs(w[k+1] - w[k]) - np.abs(C*q)
                i += 1

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

    def test(self):
        z = self.prevertex
        w = self.vertex
        C = self.C
        err = 0
        for k in range(1, len(z)):
            if np.isinf(z[k]):
                r = np.abs(w[k+1] - w[k-1]
                           - C * self.zquad(z[k-1], z[k+1], k-1, k+1))
            else:
                r = np.abs(w[k] - self.A
                           + C * self.zquad(z[k], 0, k))
            err = max(err, r)

        return err

    def map(self, z):
        """ SC transformation of z """
        return self.A + self.C * self.zquad(0,z)

    def invmap(self, w):
        """ inverse SC transformation of w """
        c = (w - self.A)/self.C

        s = solve_ivp(lambda t,z: c/self.zprod(z), [0,1], [0j])

        return newton(lambda z: self.zquad(0,z) - c, s.y[0,-1],
                      lambda z: self.zprod(z))

    def plot(self, r, theta, *arg, **kwarg):
        """
        r = radius of circles in disk
        theta = direction of rays in disk
        arg = arguments passed to plt.plot
        kwarg = keyword arguments passed to plt.plot
        """
        N = 256

        if len(r):
            r,th = np.meshgrid(r, np.linspace(0, 2*np.pi, N))
            w = self.map(r * np.exp(th * 1j))
            plt.plot(np.real(w), np.imag(w), *arg, **kwarg)

        if len(theta):
            r,th = np.meshgrid(np.linspace(0,1,N), theta)
            w = self.map(r.T * np.exp(th.T * 1j))
            plt.plot(np.real(w), np.imag(w), *arg, **kwarg)
