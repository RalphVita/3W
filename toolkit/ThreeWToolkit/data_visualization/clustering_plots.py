from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import dendrogram

from .base_visualizer import BaseVisualizer


class DataQualityHeatmap(BaseVisualizer):
    """Heatmap of quality metrics (NaN and frozen ratios) per instance × variable.

    Expects a DataFrame where rows are instances and columns are sensor variables,
    with cell values being a quality score in [0, 1] (e.g., NaN ratio, frozen ratio,
    or a combined metric).
    """

    def __init__(
        self,
        quality_df: pd.DataFrame,
        title: str = "Data Quality Heatmap",
        figsize: tuple[int, int] = (12, 8),
    ) -> None:
        self.quality_df = quality_df
        self.title = title
        self.figsize = figsize

    def plot(self, ax: Axes | None = None) -> tuple[Figure, Axes]:
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = cast(Figure, ax.get_figure())

        sns.heatmap(
            self.quality_df,
            ax=ax,
            cmap="YlOrRd",
            vmin=0.0,
            vmax=1.0,
            cbar_kws={"label": "Quality Score (0 = clean, 1 = bad)"},
            yticklabels=False,
        )
        ax.set_title(self.title)
        ax.set_xlabel("Sensor Variable")
        ax.set_ylabel("Instance")
        fig.tight_layout()
        return fig, ax


class DendrogramPlot(BaseVisualizer):
    """Dendrogram of the hierarchical clustering tree with an optional threshold cut.

    The threshold cut is drawn as a horizontal dashed red line and corresponds to the
    normalized distance at which the main cluster is extracted.
    """

    def __init__(
        self,
        linkage_matrix: np.ndarray,
        threshold: float | None = None,
        title: str = "Hierarchical Clustering Dendrogram",
        figsize: tuple[int, int] = (14, 6),
    ) -> None:
        self.linkage_matrix = linkage_matrix
        self.threshold = threshold
        self.title = title
        self.figsize = figsize

    def plot(self, ax: Axes | None = None) -> tuple[Figure, Axes]:
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = cast(Figure, ax.get_figure())

        color_threshold = self.threshold if self.threshold is not None else 0.0
        dendrogram(
            self.linkage_matrix,
            ax=ax,
            color_threshold=color_threshold,
            above_threshold_color="gray",
            no_labels=True,
        )

        if self.threshold is not None:
            ax.axhline(
                y=self.threshold,
                color="red",
                linestyle="--",
                linewidth=1.5,
                label=f"Threshold = {self.threshold:.2f}",
            )
            ax.legend()

        ax.set_title(self.title)
        ax.set_xlabel("Instances")
        ax.set_ylabel("Normalized Distance")
        fig.tight_layout()
        return fig, ax


class ClusterSizeCurvePlot(BaseVisualizer):
    """Line plot of main cluster size versus distance threshold.

    Accepts ``common_counts`` directly from ``MultivariateConsensus.common_counts_``
    (a dict mapping each threshold to the number of surviving instances).
    """

    def __init__(
        self,
        common_counts: dict[float, int],
        title: str = "Main Cluster Size vs. Threshold",
        figsize: tuple[int, int] = (10, 5),
    ) -> None:
        self.common_counts = common_counts
        self.title = title
        self.figsize = figsize

    def plot(self, ax: Axes | None = None) -> tuple[Figure, Axes]:
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = cast(Figure, ax.get_figure())

        thresholds = list(self.common_counts.keys())
        counts = list(self.common_counts.values())

        ax.plot(
            thresholds,
            counts,
            marker="o",
            linewidth=1.5,
            markersize=4,
            color="steelblue",
        )
        ax.set_title(self.title)
        ax.set_xlabel("Normalized Distance Threshold")
        ax.set_ylabel("Cluster Size (# Instances)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig, ax


class SelectionHeatmapPlot(BaseVisualizer):
    """Binary heatmap showing which instances are selected at each distance threshold.

    Rows correspond to thresholds (low at top, high at bottom) and columns correspond
    to instances. A filled cell means the instance was selected at that threshold.

    Accepts ``selection_mask`` and ``thresholds_analyzed_`` directly from a fitted
    ``MultivariateConsensus`` instance.
    """

    def __init__(
        self,
        selection_mask: np.ndarray,
        thresholds: list[float],
        title: str = "Instance Selection Heatmap",
        figsize: tuple[int, int] = (14, 6),
    ) -> None:
        self.selection_mask = selection_mask
        self.thresholds = thresholds
        self.title = title
        self.figsize = figsize

    def plot(self, ax: Axes | None = None) -> tuple[Figure, Axes]:
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = cast(Figure, ax.get_figure())

        df = pd.DataFrame(
            self.selection_mask,
            index=[f"{t:.2f}" for t in self.thresholds],
        )
        sns.heatmap(
            df,
            ax=ax,
            cmap="Blues",
            vmin=0,
            vmax=1,
            cbar=False,
            xticklabels=False,
        )
        ax.set_title(self.title)
        ax.set_xlabel("Instance Index")
        ax.set_ylabel("Distance Threshold")
        fig.tight_layout()
        return fig, ax


class ClusteringOverlayPlot(BaseVisualizer):
    """Overlays all time series, distinguishing selected from rejected instances.

    Selected instances are drawn in blue; rejected instances are drawn in gray.
    An optional centroid series (e.g., DBA centroid) is overlaid on top.

    All series are plotted on a normalized [0, 1] time axis to accommodate
    variable-length inputs.
    """

    def __init__(
        self,
        series: list[np.ndarray],
        selected_indices: list[int],
        centroid: np.ndarray | None = None,
        title: str = "Clustering Overlay",
        figsize: tuple[int, int] = (14, 6),
    ) -> None:
        self.series = series
        self.selected_indices = selected_indices
        self.centroid = centroid
        self.title = title
        self.figsize = figsize

    def plot(self, ax: Axes | None = None) -> tuple[Figure, Axes]:
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = cast(Figure, ax.get_figure())

        selected_set = set(self.selected_indices)

        for i, ts in enumerate(self.series):
            x = np.linspace(0, 1, len(ts))
            if i in selected_set:
                ax.plot(x, ts, color="steelblue", alpha=0.35, linewidth=0.8)
            else:
                ax.plot(x, ts, color="lightgray", alpha=0.5, linewidth=0.6)

        if self.centroid is not None:
            x_centroid = np.linspace(0, 1, len(self.centroid))
            ax.plot(
                x_centroid,
                self.centroid,
                color="darkblue",
                linewidth=2.0,
                label="DBA Centroid",
                zorder=5,
            )

        legend_elements = [
            Patch(facecolor="steelblue", alpha=0.6, label=f"Selected ({len(self.selected_indices)})"),
            Patch(facecolor="lightgray", alpha=0.8, label=f"Rejected ({len(self.series) - len(self.selected_indices)})"),
        ]
        if self.centroid is not None:
            from matplotlib.lines import Line2D
            legend_elements.append(Line2D([0], [0], color="darkblue", linewidth=2, label="DBA Centroid"))

        ax.legend(handles=legend_elements)
        ax.set_title(self.title)
        ax.set_xlabel("Normalized Time")
        ax.set_ylabel("Value")
        fig.tight_layout()
        return fig, ax


class RankedDistancePlot(BaseVisualizer):
    """Bar chart of elimination distances ordered by divisive rank.

    Bars are arranged from the most extreme outlier (left, rank 0) to the tightest
    centroid (right). Selected instances are shown in blue; rejected in salmon.

    Accepts ``ranking_`` and ``elimination_distances_`` directly from a fitted
    ``DivisiveRanker``, plus the ``selected_indices`` from ``MultivariateConsensus``.
    """

    def __init__(
        self,
        ranking: list[int],
        elimination_distances: list[float],
        selected_indices: list[int],
        title: str = "Ranked Distance Plot",
        figsize: tuple[int, int] = (14, 5),
    ) -> None:
        self.ranking = ranking
        self.elimination_distances = elimination_distances
        self.selected_indices = selected_indices
        self.title = title
        self.figsize = figsize

    def plot(self, ax: Axes | None = None) -> tuple[Figure, Axes]:
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = cast(Figure, ax.get_figure())

        selected_set = set(self.selected_indices)
        colors = [
            "steelblue" if idx in selected_set else "salmon"
            for idx in self.ranking
        ]

        ax.bar(
            range(len(self.ranking)),
            self.elimination_distances,
            color=colors,
            width=1.0,
            edgecolor="none",
        )

        legend_elements = [
            Patch(facecolor="steelblue", label="Selected"),
            Patch(facecolor="salmon", label="Rejected"),
        ]
        ax.legend(handles=legend_elements)
        ax.set_title(self.title)
        ax.set_xlabel("Rank (Outlier → Centroid)")
        ax.set_ylabel("Elimination Distance")
        fig.tight_layout()
        return fig, ax
