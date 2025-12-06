from abc import ABC, abstractmethod
from matplotlib.figure import Figure


class BaseVisualizer(ABC):
    """
    Base class for all visualization objects.

    Typical usage:
        vis = SomePlot(...)
        fig, path = vis.plot()
    """

    @abstractmethod
    def plot(self) -> tuple[Figure, str]:
        """Generate the plot and return (Figure, output_path)."""
        raise NotImplementedError
