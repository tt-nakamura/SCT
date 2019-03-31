import numpy as np
import matplotlib.pyplot as plt
from polygon import polygon
from diskmap import diskmap

def plot_cmplx(z, *a, **k):
    plt.plot(np.real(z), np.imag(z), *a, **k)

p = polygon([1, 1j, -1, -1j])
m = diskmap(p)

x,y = np.meshgrid(np.linspace(-0.8, 0.8, 9),
                  np.linspace(-0.99, 0.99, 100))

z = (x + y*1j)*(0.5 + 0.5j)
zi = z*1j
theta = np.linspace(0, 2*np.pi, 100)

plt.figure(figsize=(6.4, 3.2))

plt.subplot(1,2,1)
plt.axis('equal')
plt.axis('off')
plt.axis([-1,1,-1,1])
p.plot('r')
plot_cmplx(z, 'b')
plot_cmplx(zi, 'b')
plot_cmplx(p.vertex, '.y')

plt.subplot(1,2,2)
plt.axis('equal')
plt.axis('off')
plt.axis([-1,1,-1,1])
plot_cmplx(m.invmap(z), 'b')
plot_cmplx(m.invmap(zi.T).T, 'b')
plot_cmplx(np.exp(theta * 1j), 'r')
plot_cmplx(m.prevertex, '.y')

plt.tight_layout()
plt.savefig('fig1.eps')
plt.show()
