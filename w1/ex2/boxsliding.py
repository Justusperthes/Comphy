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

def slope_straight(x):
    H = 22.5
    W = 30
    slope_straight = -H / W
    return slope_straight

def slope_parabola(x):
    return 0.150 * x - 3

def ramp_system(t, z, slope):
    x, y = z
    dydx = slope(x)

    dxdt = np.sqrt(2 * g) * np.sqrt(abs(y0 - y)) / np.sqrt(1 + dydx**2)
    dydt = dydx * dxdt

    return [dxdt, dydt]

def stop_at_x_30(t, z):
    x, y = z
    return 30 - x

stop_at_x_30.terminal = True
stop_at_x_30.direction = -1  

# initial conditions for all ramps
y_init = 22.4999999 
x_init = 0
init_conditions = [x_init, y_init]

# solving the straight ramp
sol = solve_ivp(lambda t, z: ramp_system(t, z, slope=slope_straight), trange, init_conditions, max_step=0.05, events=stop_at_x_30)
ts_straight = sol.t
xs_straight = sol.y[0]
ys_straight = sol.y[1]
t_event = sol.t_events[0][0]

# how long does it take for the ball on the straight ramp?
print(f'Time taken to reach end of ramp: {t_event} seconds')

plt.rc('font', size=16)
fig, ax = plt.subplots(1, 1)
ax.plot(xs_straight, ys_straight, 'o', linestyle='--')
ax.grid()
ax.set_xlim([0, 40])
ax.set_ylim([-10, 30])
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
plt.savefig('lin_ramp.png')
plt.close()

#animation of straight ramp ball
box, = ax.plot([], [], 'ro', markersize=10)
ramp, = ax.plot(xs_straight, ys_straight, 'b-') 

def update(frame):
    data = np.array([[xs_straight[frame], ys_straight[frame]]])
    box.set_data([xs_straight[frame]],[ys_straight[frame]])
    return [box]

anim = animation.FuncAnimation(fig, update, frames=range(0, len(ts_straight), 2), interval=50, blit=True)
anim.save('sliding_box_straight.gif', writer=PillowWriter(fps=200))
plt.close()

# solving the parabolic ramp ball
sol = solve_ivp(lambda t, z: ramp_system(t, z, slope=slope_parabola), trange, init_conditions, max_step=0.05, events=stop_at_x_30)

ts_para = sol.t
xs_para = sol.y[0]
ys_para = sol.y[1]
t_event = sol.t_events[0][0]

# time taken for the parabolic ramp ball
print(f'Time taken to reach end of ramp: {t_event} seconds')

# png of parabolic ramp ball
plt.rc('font', size=16)
fig, ax = plt.subplots(1, 1)
ax.plot(xs_para, ys_para, 'o', linestyle='--')
ax.grid()
ax.set_xlim([0, 40])
ax.set_ylim([-10, 30])
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
plt.savefig('para_ramp.png')
plt.close()

#animation of parabolic ramp ball
box, = ax.plot([], [], 'ro', markersize=10)
ramp, = ax.plot(xs_para, ys_para, 'b-') 

def update(frame):
    data = np.array([[xs_para[frame], ys_para[frame]]])
    box.set_data([xs_para[frame]],[ys_para[frame]])
    return [box]

anim = animation.FuncAnimation(fig, update, frames=range(0, len(ts_para), 2), interval=50, blit=True)
anim.save('sliding_box_parabolic.gif', writer=PillowWriter(fps=200))
plt.close()

# both linear and parabolic png
plt.rc('font', size=16)
fig, ax = plt.subplots(1, 1)
ax.plot(xs_straight, ys_straight, 'o-', linestyle='--', label='Straight Ramp')
ax.plot(xs_para, ys_para, 's-', linestyle='-', label='Parabolic Ramp')
ax.grid()
ax.set_xlim([0, 40])
ax.set_ylim([-10, 30])
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.legend()
plt.savefig('combined_ramps.png')
plt.close()

# this is for standardizing lin and para handling
num_frames = max(len(ts_straight), len(ts_para))
xs_straight_interp = np.interp(np.linspace(0, 1, num_frames), np.linspace(0, 1, len(ts_straight)), xs_straight)
ys_straight_interp = np.interp(np.linspace(0, 1, num_frames), np.linspace(0, 1, len(ts_straight)), ys_straight)
xs_para_interp = np.interp(np.linspace(0, 1, num_frames), np.linspace(0, 1, len(ts_para)), xs_para)
ys_para_interp = np.interp(np.linspace(0, 1, num_frames), np.linspace(0, 1, len(ts_para)), ys_para)

# initialize the two ramps for animation
ramp_straight, = ax.plot(xs_straight_interp, ys_straight_interp, 'b-', label='Straight Ramp')
ramp_para, = ax.plot(xs_para_interp, ys_para_interp, 'g-', label='Parabolic Ramp')

# create the animation
plt.rc('font', size=16)
fig, ax = plt.subplots()
ax.set_xlim([0, 40])
ax.set_ylim([-10, 30])
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.grid()

# Initialize markers for both ramps
box_straight, = ax.plot([], [], 'ro', markersize=10)
box_para, = ax.plot([], [], 'bs', markersize=10)
ax.legend()

def update(frame):
    box_straight.set_data([xs_straight_interp[frame]], [ys_straight_interp[frame]])
    box_para.set_data([xs_para_interp[frame]], [ys_para_interp[frame]])
    
    return [box_straight, box_para]

anim = animation.FuncAnimation(fig, update, frames=range(0, num_frames), interval=50, blit=True)
anim.save('sliding_box_combined.gif', writer=PillowWriter(fps=200))
plt.close()