from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .base_visualizer import BaseVisualizer


class PlotFFT(BaseVisualizer):
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
        if self.series.empty:
            raise ValueError("Input series is empty")

        clean_series = self.series.dropna()
        if clean_series.empty:
            raise ValueError("Series contains only NaN values")

        N = len(clean_series)

        if self.sample_rate is None:
            T = 1.0
            freq_unit = "Cycles per Sample"
        else:
            T = 1.0 / self.sample_rate
            freq_unit = "Frequency (Hz)"

        yf = np.fft.fft(np.asarray(clean_series.values))
        xf = np.fft.fftfreq(N, T)[: N // 2]
        amplitude = 2.0 / N * np.abs(yf[0 : N // 2])

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
