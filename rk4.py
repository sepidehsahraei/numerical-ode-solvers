import numpy as np

def rk4_method(f, t_values, y0):
    """
    Runge-Kutta 4th order method.
    f: function f(t, y)
    t_values: array of time points
    y0: initial value
    """
    y_values = np.zeros(len(t_values))
    y_values[0] = y0

    for i in range(1, len(t_values)):
        h = t_values[i] - t_values[i - 1]
        k1 = f(t_values[i - 1], y_values[i - 1])
        k2 = f(t_values[i - 1] + h / 2, y_values[i - 1] + h / 2 * k1)
        k3 = f(t_values[i - 1] + h / 2, y_values[i - 1] + h / 2 * k2)
        k4 = f(t_values[i - 1] + h, y_values[i - 1] + h * k3)
        y_values[i] = y_values[i - 1] + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)

    return y_values
