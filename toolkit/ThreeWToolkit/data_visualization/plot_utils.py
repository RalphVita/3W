from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


def create_subplot_grid(
    nrows: int, ncols: int, figsize: Tuple[int, int] | None = None
) -> tuple[Figure, np.ndarray]:
    """
    Create a standardized grid of subplots with consistent styling.
    """
    if figsize is None:
        figsize = (5 * ncols, 4 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

    # Ensure axes is always a 2D array for consistent indexing
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1 or ncols == 1:
        axes = axes.reshape(nrows, ncols)

    fig.tight_layout(pad=3.0)

    return fig, axes
