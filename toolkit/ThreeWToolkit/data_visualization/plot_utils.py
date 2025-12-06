from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from ThreeWToolkit.constants import PLOTS_DIR


def save_plot(title: str) -> str:
    """
    Save the current matplotlib figure to the plots directory.

    This mirrors the old DataVisualization._save_plot behavior.
    """
    plot_dir = Path(PLOTS_DIR)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Clean the title for use as filename
    safe_title = title.replace(" ", "_").replace("/", "_").replace("\\", "_")
    safe_title = "".join(
        c for c in safe_title if c.isalnum() or c in ["_", "-"]
    ).lower()
    filename = f"{safe_title}.png"

    filepath = plot_dir / filename

    plt.savefig(
        filepath,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )

    print(f"DataVisualization: Chart saved to '{filepath}'")
    return str(filepath)


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
