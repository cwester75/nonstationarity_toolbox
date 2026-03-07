def binary_partition(x):
    return 0 if x < 0.5 else 1


def encode(traj):
    return [binary_partition(x) for x in traj]
