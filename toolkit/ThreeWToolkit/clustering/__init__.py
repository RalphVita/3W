"""Time series clustering sub-package for ThreeWToolkit.

This module provides scikit-learn compatible transformers and estimators
for analyzing the structural similarity of time series datasets, focused on
detecting and characterizing different operational behaviors.
"""

from ._consensus import MultivariateConsensus
from ._distances import DistanceComputer
from ._divisive import DivisiveRanker
from ._hierarchical import HierarchicalClusterer
from ._normalization import TimeSeriesScaler
from ._quality import InstanceQualityFilter
from ._resampling import TimeSeriesResampler

__all__ = [
    "InstanceQualityFilter",
    "TimeSeriesResampler",
    "TimeSeriesScaler",
    "DistanceComputer",
    "HierarchicalClusterer",
    "DivisiveRanker",
    "MultivariateConsensus",
]
