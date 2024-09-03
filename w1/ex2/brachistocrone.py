import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from scipy.optimize import fsolve

eps = 1e-8

# solve for phi1
def func1(phi1):
    return (np.cos(phi1)-1)/(phi1-np.sin(phi1)+eps)+3/4
phi1 = fsolve(func1,10)[0]

# brachistochrone must pass through (30,0)
x = 30
y = 0
# solve for a, y0
def func2(z):
    eq1 = z[0] * (phi1 - np.sin(phi1)) - x
    eq2 = z[0] * (1 + np.cos(phi1)) + z[1] - y
    return [eq1, eq2]
a_y0 = fsolve(func2, [100, 10])
a = a_y0[0]
y0 = a_y0[1]

# parametric equation
def x(phi):
    return a * (phi - np.sin(phi))
def y(phi):
    return a * (1 + np.cos(phi)) + y0

# generate range of phi values
phi = np.linspace(0, phi1, 1000)  # Adjust the range as needed

x_values = x(phi)
y_values = y(phi)

# plot parametric curve
plt.plot(x_values, y_values, label=r'$x(\phi), y(\phi)$')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Parametric Plot of x = x(φ), y = y(φ)')
plt.legend()
plt.grid(True)
plt.savefig('brachistochrone.png')
plt.close()

# y and dydx as functions of x
def func(x):
    
    return y

