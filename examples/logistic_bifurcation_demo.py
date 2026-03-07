import numpy as np
import matplotlib.pyplot as plt
from nonlinear.bifurcation.bifurcation_diagram import generate_bifurcation

r = np.linspace(2.5, 4, 2000)

rs, xs = generate_bifurcation(r)

plt.plot(rs, xs, ',')
plt.xlabel('r')
plt.ylabel('x')
plt.title('Logistic Map Bifurcation')
plt.show()
