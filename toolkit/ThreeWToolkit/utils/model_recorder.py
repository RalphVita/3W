from pathlib import Path
from typing import Any
from torch.nn import Module
from sklearn.base import BaseEstimator

class ModelRecorder:
    @staticmethod
    def save_best_model(model: Module | BaseEstimator, filename: str | Path) -> None:
        """
        Save a model to disk depending on its type and file extension.
        Supports PyTorch and scikit-learn (Pickle).

        Parameters:
            model: Trained model object.
            filename: File name where the model will be saved.
        """
        if isinstance(filename, (str, Path)):
            path = Path(filename)
            ext = path.suffix.lower()
        else:
            raise TypeError(
                f"Invalid filename: `{filename}`. Expected a string or path-like object."
            )

        # PyTorch
        if ext in [".pt", ".pth"]:
            try:
                import torch

                torch.save(model.state_dict(), filename)
            except Exception as e:
                raise RuntimeError(f"Error saving PyTorch model: {e}")

        # scikit-learn or generic Python object
        elif ext in [".pkl", ".pickle"]:
            try:
                import pickle

                with open(filename, "wb") as f:
                    pickle.dump(model, f)
            except Exception as e:
                raise RuntimeError(f"Error saving Pickle model: {e}")

        else:
            raise ValueError(f"Unsupported file extension: {ext}")

    @staticmethod
    def load_model(filename: str | Path, model: Module | None = None) -> Module | dict | BaseEstimator:
        """
        Load a model from disk depending on its type and file extension.
        Supports PyTorch (.pt, .pth) and scikit-learn/Pickle (.pkl, .pickle).

        Parameters:
            filename (str | Path):  Path pointing to the saved model file.
            model (Module, optional):  An uninitialized model instance to load weights into. Required for PyTorch models.
        """
        if isinstance(filename, (str, Path)):
            path = Path(filename)
            ext = path.suffix.lower()
        else:
            raise TypeError(
                f"Invalid filename: `{filename}`. Expected a string or path-like object."
            )

        # PyTorch
        if ext in [".pt", ".pth"]:
            try:
                import torch

                if model is None:
                    # Returns only the state_dict in case it does not have the model
                    return torch.load(filename)
                state_dict = torch.load(filename)
                model.load_state_dict(state_dict)
                return model
            except Exception as e:
                raise RuntimeError(f"Error loading PyTorch model: {e}")

        # scikit-learn or generic using Pickle
        elif ext in [".pkl", ".pickle"]:
            try:
                import pickle

                with open(filename, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                raise RuntimeError(f"Error loading Pickle model: {e}")

        else:
            raise ValueError(f"Unsupported file extension: {ext}")
