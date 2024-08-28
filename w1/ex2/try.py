import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import PillowWriter
plt.rc('font', size=16)

N = 101
xs = np.linspace(0,90,N)
ys = (100-((xs+10)%20-10)**2) * np.exp(-xs/30)
xs = xs * np.exp(-xs/200)

fig, ax = plt.subplots(figsize=(4, 6))
ax.axis('equal')
ax.plot(xs,ys)
plt.savefig('try.png')
plt.close()


plt.rc('animation', html='jshtml')

ball = ax.plot([],[],'ro')[0]

def update(i):
    ball.set_data([xs[i]], [ys[i]])
    return [ball]


anim = animation.FuncAnimation(fig,
                               update,
                               frames=N,
                               interval=10,
                               blit=True)
anim.save('try.gif', writer=PillowWriter(fps=10))
plt.close()
