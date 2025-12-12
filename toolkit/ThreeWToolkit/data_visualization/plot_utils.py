from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


def create_subplot_grid(
    nrows: int,
    ncols: int,
    figsize: Tuple[int, int] | None = None,
    default_width_per_col: int = 5,
    default_height_per_row: int = 4,
) -> tuple[Figure, np.ndarray]:

    """
    Create a grid of subplots with consistent sizing and layout.

    Parameters
    ----------
    nrows : int
        Number of rows in the subplot grid.
    ncols : int
        Number of columns in the subplot grid.
    figsize : tuple(int, int) or None
        Figure size (width, height). If None, a default based on grid size is used.

    Returns
    -------
    fig : Figure
        The created matplotlib Figure.
    axes : np.ndarray
        A 2D array of Axes objects for indexing subplots uniformly.
    """

    if figsize is None:
        figsize = (default_width_per_col  * ncols, default_height_per_row * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

    # Ensure axes is always a 2D array for consistent indexing
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1 or ncols == 1:
        axes = axes.reshape(nrows, ncols)

    fig.tight_layout(pad=3.0)

    return fig, axes
