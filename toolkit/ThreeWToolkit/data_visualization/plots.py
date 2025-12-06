from abc import ABC
from matplotlib.figure import Figure
import numpy as np
from matplotlib.axes import Axes
import pandas as pd

from .plot_utils import save_plot, create_subplot_grid
from .plot_series import PlotSeries
from .plot_multiple_series import PlotMultipleSeries
from .correlation_heatmap import CorrelationHeatmap
from .plot_fft import PlotFFT
from .seasonal_decomposition import SeasonalDecompositionPlot
from .wavelet_spectrogram import WaveletSpectrogramPlot


class DataVisualization(ABC):
    """
    Façade class exposing the old static API.

    - New style:
        vis = PlotSeries(...)
        fig, path = vis.plot()

    - Old style (still supported):
        fig, path = DataVisualization.plot_series(...)
    """

    @staticmethod
    def plot_series(
        series: pd.Series,
        title: str,
        xlabel: str,
        ylabel: str,
        overlay_events: bool = False,
        ax: Axes | None = None,
        **plot_kwargs,
    ) -> tuple[Figure, str]:
        vis = PlotSeries(
            series=series,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            overlay_events=overlay_events,
            ax=ax,
            **plot_kwargs,
        )
        return vis.plot()

    @staticmethod
    def plot_multiple_series(
        series_list: list[pd.Series],
        labels: list[str],
        title: str,
        xlabel: str,
        ylabel: str,
        ax: Axes | None = None,
        **plot_kwargs,
    ) -> tuple[Figure, str]:
        vis = PlotMultipleSeries(
            series_list=series_list,
            labels=labels,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            ax=ax,
            **plot_kwargs,
        )
        return vis.plot()

    @staticmethod
    def correlation_heatmap(
        df_of_series: pd.DataFrame,
        ax: Axes | None = None,
        **kwargs,
    ) -> tuple[Figure, str]:
        vis = CorrelationHeatmap(df_of_series=df_of_series, ax=ax, **kwargs)
        return vis.plot()

    @staticmethod
    def plot_fft(
        series: pd.Series,
        title: str = "FFT Analysis",
        sample_rate: float | None = None,
    ) -> tuple[Figure, str]:
        vis = PlotFFT(series=series, title=title, sample_rate=sample_rate)
        return vis.plot()

    @staticmethod
    def seasonal_decompose(
        series: pd.Series,
        model: str = "additive",
        period: int | None = None,
    ) -> tuple[Figure, str]:
        vis = SeasonalDecompositionPlot(series=series, model=model, period=period)
        return vis.plot()

    @staticmethod
    def plot_wavelet_spectrogram(
        series: pd.Series,
        title: str = "Wavelet Spectrogram",
    ) -> tuple[Figure, str]:
        vis = WaveletSpectrogramPlot(series=series, title=title)
        return vis.plot()

    # ------------------------------------------------------------------
    # Utilities (backwards compatible)
    # ------------------------------------------------------------------

    @staticmethod
    def _save_plot(title: str) -> str:
        """
        Backwards-compatible wrapper around plot_utils.save_plot().
        """
        return save_plot(title)

    @staticmethod
    def create_subplot_grid(
        nrows: int, ncols: int, figsize: tuple[int, int] | None = None
    ) -> tuple[Figure, np.ndarray]:
        """
        Backwards-compatible wrapper around plot_utils.create_subplot_grid().
        """
        return create_subplot_grid(nrows=nrows, ncols=ncols, figsize=figsize)
