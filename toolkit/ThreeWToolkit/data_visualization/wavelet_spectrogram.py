import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from .base_visualizer import BaseVisualizer
from .plot_utils import save_plot


class WaveletSpectrogramPlot(BaseVisualizer):
    """
    Mock wavelet spectrogram visualization.
    """

    def __init__(
        self,
        series: pd.Series,
        title: str = "Wavelet Spectrogram",
    ) -> None:
        self.series = series
        self.title = title

    def plot(self) -> tuple[Figure, str]:
        if self.series.empty:
            raise ValueError("Input series is empty")

        fig, ax = plt.subplots(figsize=(12, 8))

        time_points = len(self.series)
        frequency_scales = 50
        mock_spectrogram = np.random.rand(frequency_scales, time_points)

        for i in range(frequency_scales):
            mock_spectrogram[i, :] *= np.exp(-i / frequency_scales * 2)

        extent_tuple = (0.0, float(time_points), 1.0, float(frequency_scales))
        im = ax.imshow(
            mock_spectrogram,
            aspect="auto",
            cmap="inferno",
            origin="lower",
            extent=extent_tuple,
        )

        ax.set_title(self.title, fontsize=14)
        ax.set_xlabel("Time Index")
        ax.set_ylabel("Frequency Scale")

        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Magnitude", rotation=270, labelpad=15)

        ax.text(
            0.02,
            0.98,
            "Note: Mock implementation",
            transform=ax.transAxes,
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
            verticalalignment="top",
        )

        plt.tight_layout()

        img_path = save_plot(self.title)
        return fig, img_path
