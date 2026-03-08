import numpy as np


def arnold_cat_map(image, n=1):
    """Apply Arnold's cat map to an image n times.

    image: 2-D numpy array (square).
    """
    N = image.shape[0]
    result = image.copy()

    for _ in range(n):
        new = np.zeros_like(result)
        for x in range(N):
            for y in range(N):
                nx = (2 * x + y) % N
                ny = (x + y) % N
                new[nx, ny] = result[x, y]
        result = new

    return result


def baker_map_image(image, n=1):
    """Apply baker's map to a square image n times."""
    N = image.shape[0]
    result = image.copy()

    for _ in range(n):
        new = np.zeros_like(result)
        half = N // 2
        for x in range(N):
            for y in range(N):
                if x < half:
                    nx = 2 * x
                    ny = y // 2
                else:
                    nx = 2 * (x - half) + 1
                    ny = (y + N) // 2
                nx = nx % N
                ny = ny % N
                new[nx, ny] = result[x, y]
        result = new

    return result


def recurrence_period(image, map_func, max_iter=1000, tol=0):
    """Find the recurrence period of an image under a map."""
    original = image.copy()
    current = image.copy()

    for i in range(1, max_iter + 1):
        current = map_func(current, n=1)
        if np.sum(np.abs(current.astype(float) - original.astype(float))) <= tol:
            return i

    return None
