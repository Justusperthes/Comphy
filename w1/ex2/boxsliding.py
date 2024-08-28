import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

plt.rc('animation', html='jshtml') #something to do with with jupyter notebooks, I think

tinit = 0
tfinal = 10
trange = [tinit, tfinal]
g = 9.8
y0 = 22.5

H = 22.5
W = 30
dydx = H / W  # slope

def ramp_system(t, z):
    x, y = z
    if y > y0:
        y = y0  
    if y < 0:
        y = 0

    dxdt = np.sqrt(2 * g) * np.sqrt(max(y0 - y, 0)) / np.sqrt(1 + dydx**2)
    dydt = -abs(dydx * dxdt)  

    return [dxdt, dydt]

def stop_at_x_30(t, z):
    x, y = z
    return 30 - x

stop_at_x_30.terminal = True
stop_at_x_30.direction = -1  

y_init = 22.4999999 
x_init = 30 - y_init/dydx
print(x_init)
init_conditions = [x_init, y_init]

sol = solve_ivp(ramp_system, trange, init_conditions, max_step=0.05, events=stop_at_x_30)
ts = sol.t
xs = sol.y[0]
ys = sol.y[1]
t_event = sol.t_events[0][0]

print(f'Time taken to reach end of ramp: {t_event} seconds')

plt.rc('font', size=16)
fig, ax = plt.subplots(1, 1)
ax.plot(xs, ys, 'o', linestyle='--')
ax.grid()
ax.set_xlim([0, 40])
ax.set_ylim([-10, 30])
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
plt.savefig('lin_ramp.png')
plt.close()

box, = ax.plot([], [], 'ro', markersize=10)
ramp, = ax.plot(xs, ys, 'b-') 

def update(frame):
    data = np.array([[xs[frame], ys[frame]]])
    box.set_data([xs[frame]],[ys[frame]])
    return [box]

anim = animation.FuncAnimation(fig, update, frames=range(0, len(ts), 2), interval=50, blit=True)
anim.save('sliding_box.gif', writer=PillowWriter(fps=200))

plt.close()