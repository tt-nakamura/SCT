import numpy as np
import matplotlib.pyplot as plt
from polygon import polygon
from diskmap import diskmap

plt.figure(figsize=(6.4, 9.6))

r = np.arange(0.1, 1, 0.07)
theta = np.arange(0, 2*np.pi, np.pi/12)

p = polygon([0, 1j, -1+1j, -1-1j, 1-1j, 1])
m = diskmap(p, -0.5-0.5j)

plt.subplot(3,2,1)
p.plot('r')
m.plot(r, theta, 'b', lw=0.5)
plt.axis('equal')
plt.axis('off')
plt.axis([-1.02, 1.02, -1.02, 1.02])

############################################################

m = diskmap(p, 0.3-0.5j)

plt.subplot(3,2,2)
p.plot('r')
m.plot(r, theta, 'b', lw=0.5)
plt.axis('equal')
plt.axis('off')
plt.axis([-1.02, 1.02, -1.02, 1.02])

############################################################

p = polygon([np.exp(1j*x) for x in 2*np.pi*np.arange(5)/5])
m = diskmap(p)

x1 = np.cos(np.pi*4/5) - 0.02
y1 = np.sin(np.pi*2/5) + 0.02

plt.subplot(3,2,3)
p.plot('r')
m.plot(r, theta, 'b', lw=0.5)
plt.axis('equal')
plt.axis('off')
plt.axis([x1, x1+2*y1, -y1, y1])

############################################################

v = [np.exp(1j*x) for x in 2*np.pi*np.arange(5)/5]
v.insert(1, (v[0]+v[1])/2 + (v[1]-v[0])*1j*0.3)
v.insert(2, v[0])
v.insert(5, (v[4]+v[5])/2 + (v[5]-v[4])*1j*0.3)
v.insert(6, v[4])

p = polygon(v)
m = diskmap(p)

plt.subplot(3,2,4)
p.plot('r')
m.plot(r, theta, 'b', lw=0.5)
plt.axis('equal')
plt.axis('off')
plt.axis([x1, x1+2*y1, -y1, y1])

############################################################

v = [np.exp(1j*x) for x in 2*np.pi*np.arange(12)/12]
v[::2] /= np.sqrt(3)

p = polygon(v)
m = diskmap(p)

plt.subplot(3,2,5)
p.plot('r')
m.plot(r, theta, 'b', lw=0.5)
plt.axis('equal')
plt.axis('off')
plt.axis([-1.02, 1.02, -1.02, 1.02])

############################################################

p = polygon([1+2j, -2+2j, -2-1j, -1-1j, -1, -1-2j,
             2-2j, 2+1j, 1+1j, 1])
m = diskmap(p)

plt.subplot(3,2,6)
p.plot('r')
m.plot(r, theta, 'b', lw=0.5)
plt.axis('equal')
plt.axis('off')
plt.axis([-2.02, 2.02, -2.02, 2.02])

plt.tight_layout()
plt.savefig('fig4.eps')
plt.show()

