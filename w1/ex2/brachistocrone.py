import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from scipy.optimize import fsolve


def func(phi1):
    eps = 1e-8
    return (np.cos(phi1)-1)/(phi1-np.sin(phi1)+eps)+3/4
root = fsolve(func,10)
print(root)