import numpy as np

def heun_method(f, t_values, y0):
    """
    Heun's method (Improved Euler) for solving ODEs.
    f: function f(t, y)
    t_values: array of time points
    y0: initial value
    """
    y_values = np.zeros(len(t_values))
    y_values[0] = y0

    for i in range(1, len(t_values)):
        h = t_values[i] - t_values[i - 1]
        k1 = f(t_values[i - 1], y_values[i - 1])
        k2 = f(t_values[i], y_values[i - 1] + h * k1)
        y_values[i] = y_values[i - 1] + (h / 2) * (k1 + k2)

    return y_values
