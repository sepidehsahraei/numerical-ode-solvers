import numpy as np
import matplotlib.pyplot as plt
from methods.euler import euler_method
from methods.midpoint import midpoint_method
from methods.heun import heun_method
from methods.rk4 import rk4_method

# Differential equation
def f(t, y):
    return -2 * y

# Analytical solution
def analytical_solution(t):
    return np.exp(-2 * t)

# Time settings
t0 = 0
y0 = 1
t_end = 5
h = 0.1

# Time array
t_values = np.arange(t0, t_end + h, h)

# Numerical methods
y_euler = euler_method(f, t_values, y0)
y_midpoint = midpoint_method(f, t_values, y0)
y_heun = heun_method(f, t_values, y0)
y_rk4 = rk4_method(f, t_values, y0)
y_exact = analytical_solution(t_values)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(t_values, y_exact, label="Analytical", linewidth=2)
plt.plot(t_values, y_euler, 'o-', label="Euler", markersize=3)
plt.plot(t_values, y_midpoint, 's-', label="Midpoint", markersize=3)
plt.plot(t_values, y_heun, 'd-', label="Heun", markersize=3)
plt.plot(t_values, y_rk4, 'x-', label="RK4", markersize=3)

plt.xlabel("t")
plt.ylabel("y")
plt.title("Comparison of ODE Numerical Methods")
plt.legend()
plt.grid(True)
plt.show()
