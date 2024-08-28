import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

tinit = 0
tfinal = 10
trange = [tinit, tfinal]
g = 9.8

x_init = 0
y_init = 22.5
init = [x_init, y_init]  

H = 22.5
W = 30
dydx = H / W #slope

def ramp_system(t, z):
    x, y = z
    
    dxdt = np.sqrt(2 * g) * np.sqrt(y_init - y) / np.sqrt(1 + dydx**2)
    print(f"dxdt: {dxdt}")
    dydt = dydx * dxdt
    print(f"dydt: {dydt}")
    return [dxdt, dydt]  

mysol = solve_ivp(ramp_system, trange, init, max_step=0.01)
ts = mysol.t
xs = mysol.y[0]
ys = mysol.y[1]

plt.rc('font', size=16)
fig, ax = plt.subplots(1, 1)
ax.plot(xs, ys, 'o', linestyle='--')
ax.grid()
ax.set_xlim([0, 40])
ax.set_ylim([-10, 30])
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')

plt.savefig('lin_ramp.png')