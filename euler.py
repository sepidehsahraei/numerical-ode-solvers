import numpy as np

def euler_method(f, t_values, y0):

    y_values = np.zeros(len(t_values))
    y_values[0] = y0

    for i in range(1, len(t_values)):
        h = t_values[i] - t_values[i - 1]
        y_values[i] = y_values[i - 1] + h * f(t_values[i - 1], y_values[i - 1])

    return y_values
