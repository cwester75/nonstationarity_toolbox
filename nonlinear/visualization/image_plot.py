import numpy as np
import matplotlib.pyplot as plt


def plot_image_iterations(image, map_func, iterations, cols=4):
    """Show an image under repeated application of a chaotic map.

    Args:
        image: 2-D numpy array (square).
        map_func: function(image, n=1) -> transformed image.
        iterations: list of iteration counts to display, e.g. [0, 1, 2, 5, 10].
        cols: number of columns in the subplot grid.
    """
    rows = int(np.ceil(len(iterations) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.atleast_2d(axes)

    current = image.copy()
    prev_n = 0

    for idx, n in enumerate(iterations):
        row, col = divmod(idx, cols)
        ax = axes[row, col]

        # Advance to iteration n
        if n > prev_n:
            current = map_func(current, n=n - prev_n)
            prev_n = n

        ax.imshow(current, cmap='gray', interpolation='nearest')
        ax.set_title(f'n = {n}')
        ax.axis('off')

    # Hide unused axes
    for idx in range(len(iterations), rows * cols):
        row, col = divmod(idx, cols)
        axes[row, col].axis('off')

    fig.suptitle('Image Under Chaotic Map Iterations')
    plt.tight_layout()
    return fig, axes
