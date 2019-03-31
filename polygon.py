import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

class polygon:
    def __init__(self, w, alpha=[]):
        """
        w = vertices of polygon as complex numbers
        alpha = interior angles between edges
        """
        n = len(w)
        if n<3:
            print('there must be 3 or more vertices'); exit()

        for k in range(n):
            if np.isfinite(w[k-1]) and np.isfinite(w[k]):
                K = k
                break
        else:
            print('there must be a finite edge'); exit()

        for k in range(n):
            if np.isfinite(w[k]): continue
            if np.isinf(w[k-1]):
                print('remove consecutive infinities'); exit()
            if len(alpha)<=k:
                print('supply angles'); exit();
            if alpha[k]>0 or alpha[k]<-2:
                print('wrong angle at infinity %d' % k); exit()

        beta = np.empty(n)
        gamma = np.angle(w[K] - w[K-1])/np.pi
        EPS = 1.e-8

        for k in range(n):
            k1 = (K+k)%n
            k2 = (k1+1)%n
            if np.isinf(w[k1]) or np.isinf(w[k2]):
                beta[k1] = 1 - alpha[k1]
            else:
                beta[k1] = np.angle(w[k2] - w[k1])/np.pi - gamma
                beta[k1] = (beta[k1] + 1)%2 - 1
                if beta[k1] > 1-EPS: beta[k1] = -1

            gamma += beta[k1]

        if np.abs((np.sum(beta) + 1)%2 - 1) > EPS:
            print('angles do not add up to 2'); exit()
        if beta[K-2] == 0 or beta[K-2] == -1:
            print('cannot determine prevertex %d' % (K-2)); exit()

        self.vertex = np.array(w, dtype=np.complex)
        self.angle = beta
        self.K = K

    def FiniteEdge(self, k_range=[], N=-1):
        w = self.vertex
        n = len(w)
        if len(k_range)<2: [k,k0] = [0,n-1]
        else: [k,k0] = [k_range[0]%n, k_range[1]%n]
        if N<0: N=n-1
        e = []
        while k!=k0 and len(e)<N:
            k1 = (k+1)%n
            if np.isfinite(w[k]) and np.isfinite(w[k1]):
                e.append(k)
            k=k1

        return e

    def InfVertex(self, k_range=[], N=-1):
        n = len(self.vertex)
        if len(k_range)<2: [k,k0] = [0,n-1]
        else: [k,k0] = [k_range[0]%n, k_range[1]%n]
        if N<0: N=n-1
        e = []
        while k!=k0 and len(e)<N:
            if np.isinf(self.vertex[k]): e.append(k)
            k=(k+1)%n

        return e

    def roll(self, k=None):
        if k is None: k=-self.K
        if k==0: return
        self.vertex = np.roll(self.vertex, k)
        self.angle = np.roll(self.angle, k)
        self.K += k
        self.K %= len(self.vertex)

    def flip(self):
        self.vertex = np.flipud(self.vertex)
        self.angle = np.flipud(self.angle)
        self.K *= -1
        self.K %= len(self.vertex)

    def copy(self):
        return deepcopy(self)

    def plot(self, *arg, **kwarg):
        """
        arg = arguments passed to plt.plot
        kwarg = keyword arguments passed to plt.plot
        """
        R = 10

        w = np.roll(self.vertex, -self.K)
        beta = np.roll(self.angle, -self.K)
        gamma = np.angle(w[0] - w[-1]) + np.pi*np.cumsum(beta)
        v = [ w[-1] ]
        for k in range(len(w)):
            if np.isinf(w[k]):
                v.append(v[-1] + R*np.exp(gamma[k-1] * 1j))
                plt.plot(np.real(v), np.imag(v), *arg, **kwarg)
                v = [w[k+1] - R*np.exp(gamma[k] * 1j)]
            else:
                v.append(w[k])

        plt.plot(np.real(v), np.imag(v), *arg, **kwarg)
