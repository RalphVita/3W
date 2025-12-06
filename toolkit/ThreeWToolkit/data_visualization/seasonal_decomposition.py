import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
from statsmodels.tsa.seasonal import seasonal_decompose

from .base_visualizer import BaseVisualizer
from .plot_utils import save_plot


class SeasonalDecompositionPlot(BaseVisualizer):
    """
    Performs seasonal decomposition and plots the components.
    """

    def __init__(
        self,
        series: pd.Series,
        model: str = "additive",
        period: int | None = None,
    ) -> None:
        self.series = series
        self.model = model
        self.period = period

    def plot(self) -> tuple[Figure, str]:
        if self.series.empty:
            raise ValueError("Input series is empty")

        if self.model not in ["additive", "multiplicative"]:
            raise ValueError("Model must be either 'additive' or 'multiplicative'")

        clean_series = self.series.dropna()
        if len(clean_series) < 10:
            raise ValueError(
                "Series too short for decomposition (minimum 10 points required)"
            )

        try:
            result = seasonal_decompose(
                clean_series, model=self.model, period=self.period
            )
        except Exception as e:  # noqa: BLE001
            raise ValueError(f"Decomposition failed: {str(e)}") from e

        fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

        result.observed.plot(ax=axes[0], color="blue", linewidth=1.5)
        axes[0].set_ylabel("Observed")
        axes[0].set_title("Original Time Series")
        axes[0].grid(True, alpha=0.3)

        result.trend.plot(ax=axes[1], color="red", linewidth=1.5)
        axes[1].set_ylabel("Trend")
        axes[1].set_title("Trend Component")
        axes[1].grid(True, alpha=0.3)

        result.seasonal.plot(ax=axes[2], color="green", linewidth=1.5)
        axes[2].set_ylabel("Seasonal")
        axes[2].set_title("Seasonal Component")
        axes[2].grid(True, alpha=0.3)

        result.resid.plot(ax=axes[3], color="orange", linewidth=1.5)
        axes[3].set_ylabel("Residual")
        axes[3].set_title("Residual Component")
        axes[3].set_xlabel("Date")
        axes[3].grid(True, alpha=0.3)

        fig.suptitle(
            f"Seasonal Decomposition ({self.model.title()} Model)",
            fontsize=16,
            y=0.98,
        )

        plt.tight_layout()
        plt.subplots_adjust(top=0.94)

        img_path = save_plot("Seasonal_Decomposition")
        return fig, img_path
