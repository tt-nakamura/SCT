import numpy as np
import matplotlib.pyplot as plt
from polygon import polygon
from hplmap import hplmap

plt.figure(figsize=(6,6))

x = np.arange(-1, 4, 0.125)
y = np.arange(0.125, 2, 0.125)

p = polygon([-1, -0.5+0.5j, 0.5-0.5j, 1, np.inf],
            [0, 0, 0,  5/4, -1])
m = hplmap(p)

plt.subplot(3,2,1)
m.plot(x, y, 'b', lw=1)
p.plot('r')
plt.axis('equal')
plt.axis('off')
plt.axis([-1.5, 1.5, -0.53, 1.5])

####################################################

x = np.arange(-1, 4, 0.125)
y = np.arange(0.125, 2.25, 0.125)

p = polygon([0, 1j, 0.5j, 1+0.5j, 1, np.inf],
            [0, 0, 0, 0, 1/2, -1])
m = hplmap(p)

plt.subplot(3,2,2)
m.plot(x, y, 'b', lw=1)
p.plot('r')
plt.axis('equal')
plt.axis('off')
plt.axis([-1, 2, -0.03, 2])

####################################################

x = np.arange(-1, 2.75, 0.125)
y = np.arange(0.125, 1.75, 0.125)

p = polygon([-1.5+0.75j, 0.75j, 0, 1.5, np.inf],
            [0, 0, 0, 1/2, 0])
m = hplmap(p)

plt.subplot(3,2,3)
m.plot(x, y, 'b', lw=1)
p.plot('r')
plt.axis('equal')
plt.axis('off')
plt.axis([-1.53, 1.53, -0.03, 2])

####################################################

x = np.arange(-1, 3, 0.125)
y = np.arange(0.125, 3, 0.125)

p = polygon([-1.5+0.75j, 0.75j,  0, np.inf],
            [0, 0, 1/2, -1/2])
m = hplmap(p)

plt.subplot(3,2,4)
m.plot(x, y, 'b', lw=1)
p.plot('r')
plt.axis('equal')
plt.axis('off')
plt.axis([-1.53, 1.53, -0.03, 2])

####################################################

x = np.arange(-1, 6, 0.125)
y = np.arange(0.125, 2, 0.125)

p = polygon([0, -1+2j, 1+3**0.5*1j, 0, np.inf],
            [0, 0, 0, 1/3, -1])
m = hplmap(p)

plt.subplot(3,2,5)
m.plot(x, y, 'b', lw=1)
p.plot('r')
plt.axis('equal')
plt.axis('off')
plt.axis([-2.1, 2.1, -0.03, 2.8])

####################################################

x = np.arange(-1, 6, 0.125)
y = np.arange(0.125, 2, 0.125)

p = polygon([0, -1+2j, 0, 1+3**0.5*1j, 0, np.inf],
            [0, 0, 0, 0, 1/3, -1])
m = hplmap(p)

plt.subplot(3,2,6)
m.plot(x, y, 'b', lw=1)
p.plot('r')
plt.axis('equal')
plt.axis('off')
plt.axis([-2.1, 2.1, -0.03, 2.8])

plt.tight_layout()
plt.savefig('fig3.eps')

plt.show()
