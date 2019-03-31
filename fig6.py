import numpy as np
import matplotlib.pyplot as plt
from polygon import polygon
from extermap import extermap

plt.figure(figsize=(6.4, 12.8))

r = np.arange(0.1, 1, 0.07)
theta = 2*np.pi * np.arange(36)/36

plt.subplot(4,2,1)

s,t = np.meshgrid(r, np.linspace(0, 2*np.pi, 100))
z = s * np.exp(1j*t)
w = 0.5*(z + 1/z)
plt.plot(np.real(w), np.imag(w), 'b', lw=0.5)

s,t = np.meshgrid(np.linspace(0.01, 0.99, 100), theta)
z = s * np.exp(1j*t)
w = 0.5*(z + 1/z)
plt.plot(np.real(w.T), np.imag(w.T), 'b', lw=0.5)

plt.plot([-1,1], [0,0], 'r')
plt.axis('equal')
plt.axis('off')
plt.axis([-3,3,-3,3])

###############################################################

p = polygon([-0.5j, -2+0.5j, -0.5j, 1.5+1j])
m = extermap(p)

plt.subplot(4,2,2)
m.plot(r, theta, 'b', lw=0.5)
p.plot('r')
plt.axis('equal')
plt.axis('off')
plt.axis([-5,5,-5,5])

###############################################################

r = np.arange(0.3, 1, 0.04)
theta = 2*np.pi * np.arange(36)/36

p = polygon([1,1j,-1,-1j])
m = extermap(p)

plt.subplot(4,2,3)
m.plot(r, theta, 'b', lw=0.5)
p.plot('r')
plt.axis('equal')
plt.axis('off')
plt.axis([-2,2,-2,2])

###############################################################

p = polygon([3+1j, 1+1j, 1+3j, -1+3j, -1+1j, -3+1j, -3-1j,
             -1-1j, -1-3j, 1-3j, 1-1j, 3-1j])
m = extermap(p)

plt.subplot(4,2,4)
m.plot(r, theta, 'b', lw=0.5)
p.plot('r')
plt.axis('equal')
plt.axis('off')
plt.axis([-5,5,-5,5])

###############################################################

p = polygon([1-1j, 2-1j, 2+1j, 1+1j, 1, -1+1j,
             -2+1j, -2-1j, -1-1j, -1])
m = extermap(p)

plt.subplot(4,2,5)
m.plot(r, theta, 'b', lw=0.5)
p.plot('r')
plt.axis('equal')
plt.axis('off')
plt.axis([-3,3,-3,3])

###############################################################

p = polygon([5+2j, 5+4j, 1+4j, 1+2j,
             -1+2j, -1, -3, -3-2j, -5-2j, -5-4j, -1-4j,
             -1-2j, 1-2j, 1, 3, 3+2j])
m = extermap(p)

plt.subplot(4,2,6)
m.plot(r, theta, 'b', lw=0.5)
p.plot('r')
plt.axis('equal')
plt.axis('off')
plt.axis([-7,7,-7,7])

###############################################################

p = polygon([0, 1j, 0, -1+0.5j, 0, -1-1.5j, 0, 1-1j, 0, 1.5+0.5j])
m = extermap(p)

plt.subplot(4,2,7)
m.plot(r, theta, 'b', lw=0.5)
p.plot('r')
plt.axis('equal')
plt.axis('off')
plt.axis([-2,2,-2,2])

###############################################################

p = polygon([-1, 0, 0.5-1j, 0.5-1.5j, 0.5-1j, 1, 0.5+1j, 0])
m = extermap(p)

plt.subplot(4,2,8)
m.plot(r, theta, 'b', lw=0.5)
p.plot('r')
plt.axis('equal')
plt.axis('off')
plt.axis([-2,2,-2,2])

plt.tight_layout()
plt.savefig('fig6.eps')
plt.show()
