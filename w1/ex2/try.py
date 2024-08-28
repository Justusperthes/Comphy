from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

tinit = 0
tfinal = 20
trange = [tinit,tfinal]
yinit = [1,-1]

dydt = lambda t, y: [y[1],-y[0]-1/10*y[1]]
mysol = solve_ivp(dydt, trange, yinit)
ts = mysol.t
y0s = mysol.y[0]
y1s = mysol.y[1]

plt.rc('font', size=16)

fig, ax = plt.subplots()
ax.plot(ts,y0s,linestyle='dashed',marker='o')
ax.plot(ts,y1s,linestyle='dashed',marker='x')
ax.set_xlabel('$t$')
ax.set_ylabel('$y$')
ax.legend(['$y_0$','$y_1$'])


plt.savefig('try.png')