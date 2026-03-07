from nonlinear.core.maps import LogisticMap
from nonlinear.core.trajectory import iterate_map
from nonlinear.symbolic.symbolic_sequence import encode
from nonlinear.symbolic.entropy import shannon_entropy

m = LogisticMap(4)

traj = iterate_map(m, 0.2, 500)

seq = encode(traj)

print("Entropy:", shannon_entropy(seq))
