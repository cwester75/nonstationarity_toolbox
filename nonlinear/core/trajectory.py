def iterate_map(map_obj, x0, n):
    traj = []
    x = x0

    for _ in range(n):
        x = map_obj.f(x)
        traj.append(x)

    return traj
