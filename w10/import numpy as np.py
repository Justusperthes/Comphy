import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Constants
k_B = 1.0  # Boltzmann constant (arbitrary units)

# Function to calculate Boltzmann probabilities
def boltzmann_distribution(E, T):
    Z = np.sum(np.exp(-E / (k_B * T)))  # Partition function
    P = np.exp(-E / (k_B * T)) / Z      # Boltzmann probabilities
    return P

# Energy levels
eps = 10e-5
E_i = np.arange(0,20,0.1)  # Energy levels (0 to 19 units)

# Temperatures to compare
temperatures = [1, 5, 10]  # Example temperatures in arbitrary units

# Plotting
plt.figure(figsize=(10, 6))
for T in temperatures:
    P = boltzmann_distribution(E_i, T)
    plt.plot(E_i, P, label=f"T = {T}")

plt.xlabel("Energy Level $E_i$")
plt.ylabel("Probability $P(E_i)$")
plt.title("Boltzmann Distribution at Different Temperatures")
plt.legend()
plt.grid()
plt.show()

T=5

# Compute the integral (area under the curve)
integral, error = quad(lambda E: boltzmann_distribution(E, T), 0, np.inf)
print(f"Integral of P(E_i) over all energies: {integral}")
