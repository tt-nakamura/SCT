import numpy as np
import matplotlib.pyplot as plt
from polygon import polygon
from diskmap import diskmap

plt.figure(figsize=(6.4, 9.6))

r = np.arange(0.1, 1, 0.07)
theta = np.arange(0, 2*np.pi, np.pi/12)

p = polygon([1-0.5j, np.inf, 1j, np.inf, -1, np.inf, 1-1j],
            [3/2, -1/4, 3/2, 0, 3/2, -1/4, 1])
m = diskmap(p)

plt.subplot(3,2,1)
m.plot(r, theta, 'b', lw=0.5)
p.plot('r')
plt.axis('equal')
plt.axis('off')
plt.axis([-2.5, 2.5, -3, 2])

############################################################

p = polygon([1-1j, np.inf, 1+1j, np.inf, -0.3+1.3j,
             np.inf, -1+0.5j, np.inf, -1-1j],
            [1, -1/6, 2, -1/3, 2, -1/4, 2, -1/4, 1])
m = diskmap(p)

plt.subplot(3,2,2)
m.plot(r, theta, 'b', lw=0.5)
p.plot('r')
plt.axis('equal')
plt.axis('off')
plt.axis([-2, 2.3, -1.02, 3.28])

############################################################

p = polygon([-1j, np.inf, 0], [1/2, -1, 3/2])
m = diskmap(p, 1)

plt.subplot(3,2,3)
m.plot(r, theta, 'b', lw=0.5)
p.plot('r')
plt.axis('equal')
plt.axis('off')
plt.axis([-2, 2, -1.02, 2.98])

############################################################

p = polygon([-1j, 1-1j, np.inf, 0],
            [1/2, 2, -2, 3/2])
m = diskmap(p, 1)

plt.subplot(3,2,4)
m.plot(r, theta, 'b', lw=0.5)
p.plot('r')
plt.axis('equal')
plt.axis('off')
plt.axis([-2,3,-2,3])

############################################################

p = polygon([-2j, np.inf, 1.5-0.5j, np.inf, 2j,
              np.inf, -1+0.5j, np.inf, -3j],
            [2, -1/6, 2, -5/6, 2, -1/3, 2, -2/3, 1])
m = diskmap(p)

plt.subplot(3,2,5)
m.plot(r, theta, 'b', lw=0.5)
p.plot('r')
plt.axis('equal')
plt.axis('off')
plt.axis([-3,3,-3,3])

############################################################

p = polygon([-3-1j, -3-3j, np.inf, -3+3j, -3+1j,
              -1+1j, -3+1j, -3-1j, -2-1j],
            [1/2, 1/2, 0, 1/2, 1/2, 2, 1/2, 1/2, 2])
m = diskmap(p)

plt.subplot(3,2,6)
m.plot(r, theta, 'b', lw=0.5)
p.plot('r')
plt.axis('equal')
plt.axis('off')
plt.axis([-3.02, 3.02, -3.02, 3.02])

plt.tight_layout()
plt.savefig('fig5.eps')
plt.show()
