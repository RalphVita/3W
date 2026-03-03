import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

from ThreeWToolkit.data_visualization.clustering_plots import (
    DataQualityHeatmap,
    DendrogramPlot,
    ClusterSizeCurvePlot,
    SelectionHeatmapPlot,
    ClusteringOverlayPlot,
    RankedDistancePlot,
)


class TestDataQualityHeatmap:
    """Test suite for DataQualityHeatmap visualizer."""

    @pytest.fixture
    def quality_df(self):
        return pd.DataFrame(
            {
                "P-MON-CKP": [0.01, 0.05, 0.80],
                "T-TPT": [0.02, 0.90, 0.03],
            },
            index=["inst_0", "inst_1", "inst_2"],
        )

    def test_returns_figure_and_axes(self, quality_df):
        viz = DataQualityHeatmap(quality_df)
        fig, ax = viz.plot()

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)

    def test_uses_provided_axes(self, quality_df):
        fig, ax = plt.subplots()
        viz = DataQualityHeatmap(quality_df)
        returned_fig, returned_ax = viz.plot(ax=ax)

        assert returned_fig is fig
        assert returned_ax is ax
        plt.close(fig)

    def test_custom_title(self, quality_df):
        viz = DataQualityHeatmap(quality_df, title="Custom Title")
        fig, ax = viz.plot()

        assert ax.get_title() == "Custom Title"
        plt.close(fig)


class TestDendrogramPlot:
    """Test suite for DendrogramPlot visualizer."""

    @pytest.fixture
    def linkage_matrix(self):
        dm = np.array([
            [0.0, 1.0, 5.0, 6.0],
            [1.0, 0.0, 4.0, 5.0],
            [5.0, 4.0, 0.0, 2.0],
            [6.0, 5.0, 2.0, 0.0],
        ])
        condensed = squareform(dm)
        return linkage(condensed, method="average")

    def test_returns_figure_and_axes(self, linkage_matrix):
        viz = DendrogramPlot(linkage_matrix)
        fig, ax = viz.plot()

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)

    def test_with_threshold_line(self, linkage_matrix):
        viz = DendrogramPlot(linkage_matrix, threshold=3.0)
        fig, ax = viz.plot()

        # Check that a horizontal line was drawn
        lines = [line for line in ax.get_lines() if line.get_linestyle() == "--"]
        assert len(lines) >= 1
        plt.close(fig)

    def test_without_threshold(self, linkage_matrix):
        viz = DendrogramPlot(linkage_matrix, threshold=None)
        fig, ax = viz.plot()
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_uses_provided_axes(self, linkage_matrix):
        fig, ax = plt.subplots()
        viz = DendrogramPlot(linkage_matrix)
        returned_fig, _ = viz.plot(ax=ax)
        assert returned_fig is fig
        plt.close(fig)


class TestClusterSizeCurvePlot:
    """Test suite for ClusterSizeCurvePlot visualizer."""

    @pytest.fixture
    def common_counts(self):
        return {0.1: 2, 0.2: 3, 0.3: 4, 0.5: 5, 1.0: 5}

    def test_returns_figure_and_axes(self, common_counts):
        viz = ClusterSizeCurvePlot(common_counts)
        fig, ax = viz.plot()

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)

    def test_plot_has_data(self, common_counts):
        viz = ClusterSizeCurvePlot(common_counts)
        fig, ax = viz.plot()
        assert ax.has_data()
        plt.close(fig)

    def test_uses_provided_axes(self, common_counts):
        fig, ax = plt.subplots()
        viz = ClusterSizeCurvePlot(common_counts)
        returned_fig, _ = viz.plot(ax=ax)
        assert returned_fig is fig
        plt.close(fig)


class TestSelectionHeatmapPlot:
    """Test suite for SelectionHeatmapPlot visualizer."""

    @pytest.fixture
    def selection_data(self):
        mask = np.array([
            [1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0],
        ])
        thresholds = [0.3, 0.5, 0.8]
        return mask, thresholds

    def test_returns_figure_and_axes(self, selection_data):
        mask, thresholds = selection_data
        viz = SelectionHeatmapPlot(mask, thresholds)
        fig, ax = viz.plot()

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)

    def test_uses_provided_axes(self, selection_data):
        mask, thresholds = selection_data
        fig, ax = plt.subplots()
        viz = SelectionHeatmapPlot(mask, thresholds)
        returned_fig, _ = viz.plot(ax=ax)
        assert returned_fig is fig
        plt.close(fig)


class TestClusteringOverlayPlot:
    """Test suite for ClusteringOverlayPlot visualizer."""

    @pytest.fixture
    def series_data(self):
        np.random.seed(42)
        series = [np.random.randn(50) for _ in range(5)]
        selected = [0, 1, 2]
        return series, selected

    def test_returns_figure_and_axes(self, series_data):
        series, selected = series_data
        viz = ClusteringOverlayPlot(series, selected)
        fig, ax = viz.plot()

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)

    def test_with_centroid(self, series_data):
        series, selected = series_data
        centroid = np.zeros(50)
        viz = ClusteringOverlayPlot(series, selected, centroid=centroid)
        fig, ax = viz.plot()

        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_without_centroid(self, series_data):
        series, selected = series_data
        viz = ClusteringOverlayPlot(series, selected, centroid=None)
        fig, ax = viz.plot()

        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_uses_provided_axes(self, series_data):
        series, selected = series_data
        fig, ax = plt.subplots()
        viz = ClusteringOverlayPlot(series, selected)
        returned_fig, _ = viz.plot(ax=ax)
        assert returned_fig is fig
        plt.close(fig)

    def test_variable_length_series(self):
        series = [np.ones(30), np.ones(50), np.ones(100)]
        viz = ClusteringOverlayPlot(series, selected_indices=[0])
        fig, ax = viz.plot()
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestRankedDistancePlot:
    """Test suite for RankedDistancePlot visualizer."""

    @pytest.fixture
    def ranked_data(self):
        ranking = [2, 0, 1]
        distances = [19.0, 1.0, 0.0]
        selected = [0, 1]
        return ranking, distances, selected

    def test_returns_figure_and_axes(self, ranked_data):
        ranking, distances, selected = ranked_data
        viz = RankedDistancePlot(ranking, distances, selected)
        fig, ax = viz.plot()

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)

    def test_plot_has_data(self, ranked_data):
        ranking, distances, selected = ranked_data
        viz = RankedDistancePlot(ranking, distances, selected)
        fig, ax = viz.plot()
        assert ax.has_data()
        plt.close(fig)

    def test_uses_provided_axes(self, ranked_data):
        ranking, distances, selected = ranked_data
        fig, ax = plt.subplots()
        viz = RankedDistancePlot(ranking, distances, selected)
        returned_fig, _ = viz.plot(ax=ax)
        assert returned_fig is fig
        plt.close(fig)

    def test_custom_title(self, ranked_data):
        ranking, distances, selected = ranked_data
        viz = RankedDistancePlot(ranking, distances, selected, title="My Plot")
        fig, ax = viz.plot()
        assert ax.get_title() == "My Plot"
        plt.close(fig)
