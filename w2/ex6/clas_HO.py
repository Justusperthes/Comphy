# 4.1 Plot the potential of the harmonic oscillator
# with some different values for the force constant k.

import numpy as np
import matplotlib.pyplot as plt

k_B = 1
T = 300
x = np.linspace (-100,100,400)

x_range = (-15, 15)
y_range = (0,25)
spring_constants = [0.5, 1, 2]

def V(x,k):
    return 0.5 * k * x**2

def P(x,k):
    return np.sqrt(k / (2 * np.pi * k_B * T)) * np.exp(-0.5 * (k * x**2) / (k_B * T)) #prob density

# Plot probability distribution
plt.figure(figsize=(8, 6))
plt.plot(x, P(x,spring_constants[0]), label='Probability Distribution P(x)')
plt.xlabel('Displacement (x)')
plt.ylabel('Probability Density P(x)')
plt.title('Classical Probability Distribution in a Harmonic Potential')
plt.ylim((0,0.02))
plt.grid(True)
plt.legend()
plt.savefig('classical_prob_dist.png')

fig, axs = plt.subplots(3, 1, figsize=(8, 12))

for i, k in enumerate(spring_constants):
    V_values = V(x,k)
    axs[i].plot(x, V_values, label=f'k = {k}')
    axs[i].set_title(f'Potential Energy with k = {k}')
    axs[i].set_xlabel('Displacement (x)')
    axs[i].set_ylabel('Potential Energy (V(x))')
    axs[i].grid(True)
    axs[i].set_xlim(x_range)
    axs[i].set_ylim(y_range)
plt.tight_layout()
plt.savefig('HO_pot.png')
plt.close

