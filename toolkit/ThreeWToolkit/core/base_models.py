from abc import ABC

from pydantic import BaseModel, Field, field_validator

from ..core.enums import ModelTypeEnum


class ModelsConfig(BaseModel):
    model_type: ModelTypeEnum | str = Field(..., description="Type of model to use.")
    random_seed: int | None = Field(42, description="Random seed for reproducibility.")

    @field_validator("model_type")
    @classmethod
    def check_model_type(cls, v, info):
        valid_types = {e for e in ModelTypeEnum}
        valid_strs = {e.value for e in ModelTypeEnum}
        if v is None:
            raise ValueError("model_type is required.")
        if v not in valid_types and v not in valid_strs:
            raise NotImplementedError(f"`model_type` {v} not implemented yet.")
        return v


class BaseModels(ABC):
    def __init__(self, config: ModelsConfig):
        """
        Base model class constructor.

        Args:
            config (ModelsConfig): Configuration object with model parameters.
        """
        super().__init__()
        self.config = config
