from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .base_visualizer import BaseVisualizer
from .plot_utils import save_plot


class PlotSeries(BaseVisualizer):
    def __init__(
        self,
        series: pd.Series,
        title: str,
        xlabel: str,
        ylabel: str,
        overlay_events: bool = False,
        ax: Axes | None = None,
        **plot_kwargs,
    ) -> None:
        self.series = series
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.overlay_events = overlay_events
        self.ax = ax
        self.plot_kwargs = plot_kwargs

    def plot(self) -> tuple[Figure, str]:
        if self.ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        else:
            ax = self.ax
            fig = cast(Figure, ax.figure)

        ax.plot(
            self.series.index,
            np.asarray(self.series.values),
            label="Value",
            **self.plot_kwargs,
        )
        ax.set_title(self.title)
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        ax.grid(True, alpha=0.3)

        if self.overlay_events:
            nan_dates = self.series.index[self.series.isna()]
            for date in nan_dates:
                ax.axvline(x=date, color="red", linestyle="--", alpha=0.7, linewidth=1)

            if len(nan_dates) > 0:
                ax.axvline(
                    x=nan_dates[0],
                    color="red",
                    linestyle="--",
                    alpha=0.7,
                    linewidth=1,
                    label="Missing Data",
                )
                ax.legend()

        img_path = save_plot(self.title)
        return fig, img_path
