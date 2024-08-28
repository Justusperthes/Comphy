import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

plt.rc('animation', html='jshtml') #something to do with with jupyter notebooks, I think

N = 50 #number of points
highlighted_dots = 5
delta_d = 1 #magnitude of DeltaP
steps = 100 #number of iterations in walk

# initial 2d array at origin
P_start = np.zeros((N,2))

# array storing rms distances at each step
rms_distances = np.zeros(steps + 1)

rms_distances[0] = 0 #walkers start at origin

# coloring and highlighting scatterplot
N_blue = N-highlighted_dots  # number of blue colors
N_red = highlighted_dots    # number of red colors
blue_color = (0, 0, 1)  
red_color = (1, 0, 0)   
colors = [blue_color] * N_blue + [red_color] * N_red

# positions in time
""" [[[x11,y11],[x12,y12],[x13,y13],...,[x1steps,y1steps]],
    [[x21,y21],[x22,y22],[x23,y23],...,[x2steps,y2steps]],
    ...
    [[xN1,yN1],[xN2,yN2],...[xNsteps,yNsteps]] ] """
points_in_time = []

# whatever needs to be done at step j
for j in range(1, steps + 1):
    # array of angles
    random_number = np.random.uniform()
    low, high = 0, 2*np.pi
    random_number_range = np.random.uniform(low,high)
    angles = np.random.uniform(low,high,N)

    # 2d array of DeltaP at different angles
    cos_values = np.cos(angles)
    sin_values = np.sin(angles)
    DeltaP = np.column_stack((cos_values*delta_d, sin_values*delta_d))
    P_end = P_start + DeltaP

    #RMS distance for each step
    distances = np.sqrt(P_end[:, 0]**2+P_end[:, 1]**2)

    #distances of each walker from origin
    rms_distances[j] = np.sqrt(np.mean(distances**2))

    #init list to store positions for the step j
    points = []

    # at a particular step (j), do the stuff to every walker (i)
    for i in range(N):
        start = P_start[i, 0]
        end = P_end[i, 0]
        x_values = [start,end]
        y_values = [start,end]
        plt.plot(x_values, y_values, marker='o')
        points.append([P_end[i, 0], P_end[i, 1]])
    
    
    points_in_time.append(points)

    P_start = P_end

# plot1.png
plt.title(f"Paths of {N} {steps}-step walks from origin")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.grid(True)
plt.savefig('plot1.png')
plt.close()

# plot2.png
plt.figure()
plt.scatter(P_end[:,0], P_end[:,1], c = colors, marker='o')
plt.title(f"End points of {N} {steps}-step walks from origin")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.grid(True)
plt.savefig('plot2.png')
plt.close()

# plot3: RMS distance vs time
plt.figure()
plt.loglog(range(steps + 1), rms_distances, marker='o')
plt.title("RMS distance from origin vs time steps")
plt.xlabel("Time Steps")
plt.ylabel("RMS Distance")
plt.grid(True)
plt.savefig('rms_distance.png')
plt.close()

# create figure for animation
fig, ax = plt.subplots()
ax.set_xlim(-steps, steps)
ax.set_ylim(-steps, steps)
ax.set_title(f"Evolution of {N} Random Walkers in {steps} steps")
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")

# init scatter plot without any data
scat = ax.scatter([], [], marker='o')

def update(frame):
    data = np.array(points_in_time[frame])
    scat.set_offsets(data)
    scat.set_color(colors)

    ax.set_xlim(-15, 15)
    ax.set_ylim(-15, 15)
    return [scat]

# create animation
anim = animation.FuncAnimation(fig, update, frames=range(steps), interval=100, blit=True)

# GIF
anim.save('random_walk_animation.gif', writer=PillowWriter(fps=10))

plt.close()