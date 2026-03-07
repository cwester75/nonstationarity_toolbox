import numpy as np
import matplotlib.pyplot as plt


def cobweb(map_obj, x0, n=50):
    xs = [x0]

    for _ in range(n):
        xs.append(map_obj.f(xs[-1]))

    x = np.linspace(0, 1, 500)

    plt.plot(x, [map_obj.f(v) for v in x])
    plt.plot(x, x)

    xn = x0

    for _ in range(n):
        yn = map_obj.f(xn)
        plt.plot([xn, xn], [xn, yn], 'r')
        plt.plot([xn, yn], [yn, yn], 'r')
        xn = yn

    plt.show()
