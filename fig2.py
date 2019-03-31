import numpy as np
import matplotlib.pyplot as plt

def plot_cmplx(z, *a, **k):
    plt.plot(np.real(z), np.imag(z), *a, **k)

def f(z):
    s = np.sqrt(z*z-1)
    s[np.imag(s)<0] *= -1
    return s + np.log(z+s)

(x1,x2,y1,y2) = (-10, 5, 0, 6)
(dx,dy) = (0.25, 0.25)

plt.figure(figsize=(6.4, 3.8))

x,y = np.meshgrid(np.linspace(x1, x2, (x2-x1)/dx+1),
                  np.linspace(y1+1e-4, y2, 256))
plot_cmplx(f(x + y*1j), 'b')

x,y = np.meshgrid(np.linspace(x1, x2, 256),
                  np.linspace(dy, y2, (y2-y1)/dy))
plot_cmplx(f(x.T + y.T*1j), 'b')

plt.plot([-10, 0, 0, 10], [np.pi, np.pi, 0, 0], 'r')
plt.axis('off')
plt.axis('equal')
plt.axis([-6,6,0,7])

plt.tight_layout()
plt.savefig('fig2.eps')
plt.show()
