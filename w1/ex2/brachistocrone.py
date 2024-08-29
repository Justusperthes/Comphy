import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from scipy.optimize import fsolve


def func1(phi1):
    eps = 1e-8
    return (np.cos(phi1)-1)/(phi1-np.sin(phi1)+eps)+3/4
root = fsolve(func1,10)[0]
print(root)

x = 30
y = 0

def func2(z):
    eq1 = z[0] * (root - np.sin(root)) - x
    eq2 = z[0] * (1 + np.cos(root)) + z[1] - y
    return [eq1, eq2]
root = fsolve(func2, [100, 10])
print(root)

# def func(x):
#     return [x[0] * np.cos(x[1]) - 4,
#             x[1] * x[0] - x[1] - 5]
# root = fsolve(func, [1, 1])
# print(root)