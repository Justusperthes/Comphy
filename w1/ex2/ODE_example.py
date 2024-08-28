import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

tinit = 0
tfinal = 3
trange = [tinit,tfinal]

yinit = [1]

dydt = lambda t, y: 0.9 * y**2 * np.cos(t)

mysol = solve_ivp(dydt, trange, yinit)
ts = mysol.t
ys = mysol.y[0]

plt.rc('font', size=16)
fig,ax = plt.subplots(1,1)
ax.plot(ts,ys,'o',linestyle='--')
ax.grid()
ax.set_ylim([0,11])
ax.set_xlabel('$t$')
ax.set_ylabel('$y$')

plt.savefig('single_ODE.png')

ts = np.linspace(tinit, tfinal, 100)
mysol = solve_ivp(dydt,
                  trange,
                  yinit,
                  t_eval=ts)
ts = mysol.t
ys = mysol.y[0]

ax.plot(ts,ys)
plt.savefig('single_ODE_w_solve_ivp.png')
fig