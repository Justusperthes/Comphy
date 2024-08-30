from scipy.integrate import quad

# Define a function to integrate
def f(x):
    return x**2

# Use quad to integrate f(x) from 0 to 1
result, error = quad(f, 0, 1)

print(f"Result: {result}, Error: {error}")
