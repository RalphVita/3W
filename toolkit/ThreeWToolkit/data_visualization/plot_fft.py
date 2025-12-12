from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .base_visualizer import BaseVisualizer


class PlotFFT(BaseVisualizer):
    """
    Visualizer for computing and plotting the Fast Fourier Transform (FFT)
    of a time series. Supports optional sample rate for frequency scaling.
    """
    def __init__(
        self,
        series: pd.Series,
        title: str = "FFT Analysis",
        sample_rate: float | None = None,
    ) -> None:
        self.series = series
        self.title = title
        self.sample_rate = sample_rate

    def plot(self, ax: Axes | None = None) -> tuple[Figure, Axes]:
        """
        Plot the FFT amplitude spectrum of the input series.

        If no Axes is provided, a new figure is created.
        Raises an error if the series is empty or contains only NaN values.

        Parameters
        ----------
        ax : Axes or None
            Axes to draw the plot on. If None, a new figure/Axes is created.

        Returns
        -------
        fig : Figure
            The figure containing the FFT plot.
        ax : Axes
            The axes with the FFT amplitude spectrum.
        """
        if self.series.empty:
            raise ValueError("Input series is empty")

        clean_series = self.series.dropna()
        if clean_series.empty:
            raise ValueError("Series contains only NaN values")

        num_samples = len(clean_series)

        if self.sample_rate is None:
            sample_period = 1.0
            freq_unit = "Cycles per Sample"
        else:
            sample_period = 1.0 / self.sample_rate
            freq_unit = "Frequency (Hz)"

        yf = np.fft.fft(clean_series.values)
        xf = np.fft.fftfreq(num_samples, sample_period)[: num_samples // 2]
        amplitude = 2.0 / num_samples * np.abs(yf[0 : num_samples // 2])

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 8))
        else:
            fig = cast(Figure, ax.get_figure())

        ax.plot(xf, amplitude, linewidth=1.5)
        ax.grid(True, alpha=0.3)
        ax.set_title(self.title)
        ax.set_xlabel(freq_unit)
        ax.set_ylabel("Amplitude")
        ax.set_xlim(0.0, float(xf.max()))

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig.tight_layout()

        return fig, ax
