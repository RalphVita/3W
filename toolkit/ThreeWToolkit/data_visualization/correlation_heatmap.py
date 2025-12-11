from typing import cast

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .base_visualizer import BaseVisualizer


class CorrelationHeatmap(BaseVisualizer):
    def __init__(
        self,
        df_of_series: pd.DataFrame,
        **kwargs,
    ) -> None:
        self.df_of_series = df_of_series
        self.kwargs = kwargs

    def plot(self, ax: Axes | None = None) -> tuple[Figure, Axes]:
        title: str = self.kwargs.pop("title", "Correlation Heatmap")
        figsize: tuple = self.kwargs.pop("figsize", (10, 8))

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = cast(Figure, ax.get_figure())

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
            return fig, ax

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

        return fig, ax
