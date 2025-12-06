from typing import cast

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .base_visualizer import BaseVisualizer
from .plot_utils import save_plot


class CorrelationHeatmap(BaseVisualizer):
    def __init__(
        self,
        df_of_series: pd.DataFrame,
        ax: Axes | None = None,
        **kwargs,
    ) -> None:
        self.df_of_series = df_of_series
        self.ax = ax
        self.kwargs = kwargs

    def plot(self) -> tuple[Figure, str]:
        title: str = self.kwargs.pop("title", "Correlation Heatmap")
        figsize: tuple = self.kwargs.pop("figsize", (10, 8))

        if self.ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            ax = self.ax
            fig = cast(Figure, ax.figure)

        if self.df_of_series.empty:
            ax.text(
                0.5,
                0.5,
                "Empty DataFrame\nNo data to display",
                fontsize=14,
                ha="center",
                va="center",
                transform=ax.transAxes,
                bbox=dict(boxstyle="round", facecolor="lightgray"),
            )
            ax.set_title(title)
            fig.tight_layout()
            return fig, save_plot(title)

        heatmap_defaults = {
            "annot": True,
            "fmt": ".2f",
            "cmap": "coolwarm",
            "center": 0,
            "square": True,
            "cbar_kws": {"shrink": 0.8},
        }
        for key, value in heatmap_defaults.items():
            self.kwargs.setdefault(key, value)

        corr_matrix = self.df_of_series.corr()
        sns.heatmap(corr_matrix, ax=ax, **self.kwargs)
        ax.set_title(title)
        fig.tight_layout()

        img_path = save_plot(title)
        return fig, img_path
