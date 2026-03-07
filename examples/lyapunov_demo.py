import numpy as np
import matplotlib.pyplot as plt
from nonlinear.core.maps import LogisticMap
from nonlinear.core.trajectory import iterate_map
from nonlinear.diagnostics.lyapunov import lyapunov_exponent

rs = np.linspace(2.5, 4, 400)

vals = []

for r in rs:
    m = LogisticMap(r)
    traj = iterate_map(m, 0.5, 1000)
    vals.append(lyapunov_exponent(m, traj))

plt.plot(rs, vals)
plt.axhline(0, color='black')
plt.title('Lyapunov Exponent')
plt.show()
