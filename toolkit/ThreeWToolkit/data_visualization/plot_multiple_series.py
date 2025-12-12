from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .base_visualizer import BaseVisualizer


class PlotMultipleSeries(BaseVisualizer):
    """
    Visualizer for plotting multiple time series on the same axes.

    Receives a list of Series and corresponding labels, and renders them
    together using a consistent color palette.
    """

    def __init__(
        self,
        series_list: list[pd.Series],
        labels: list[str],
        title: str,
        xlabel: str,
        ylabel: str,
        **plot_kwargs,
    ) -> None:
        if not series_list:
            raise ValueError("series_list must not be empty")
        if len(series_list) != len(labels):
            raise ValueError("series_list and labels must have the same length")

        self.series_list = series_list
        self.labels = labels
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.plot_kwargs = plot_kwargs

    def plot(self, ax: Axes | None = None) -> tuple[Figure, Axes]:
        """
        Plot multiple time series on a single Axes.

        If no Axes is provided, a new figure is created.
        Series that contain only NaN values are skipped.

        Parameters
        ----------
        ax : Axes or None
            Axes to draw the plot on. If None, a new figure/Axes is created.

        Returns
        -------
        fig : Figure
            The figure containing the plot.
        ax : Axes
            The axes with the rendered time series.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        else:
            fig = cast(Figure, ax.get_figure())

        cmap = plt.get_cmap("Set1", len(self.series_list))
        colors = [cmap(i) for i in range(len(self.series_list))]

        plotted_any = False
        for i, (series, label) in enumerate(zip(self.series_list, self.labels, strict=True)):
            clean_series = series.dropna()
            if clean_series.empty:
                continue
            
            ax.plot(
                series.index,
                series.values,
                label=label,
                color=colors[i],
                **self.plot_kwargs,
            )
            plotted_any = True

        if plotted_any:
            ax.legend(loc="best")

        ax.set_title(self.title)
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        ax.grid(True, alpha=0.3)

        return fig, ax
