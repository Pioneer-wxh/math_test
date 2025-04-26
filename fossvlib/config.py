
from typing import List, Optional
from dataclasses import dataclass, field
from transformers import PretrainedConfig, TrainingArguments


class FOSSVConfig(PretrainedConfig):
    """
    Configuration class for FOSSVModel.

    This class extends PretrainedConfig to include FOSSV-specific parameters.

    Attributes:
        base_model_name_or_path (str): Name or path of the model to be wrapped.
        rank (int): Rank of the FOSSV adapters.
        alpha (Optional[float]): Scaling factor for the FOSSV update. Not mentioned in the paper, works same with LoRA.
        dropout (float): Dropout probability for FOSSV matrices. Not mentioned in the paper, works same with LoRA.
        target_modules (List[str]): Names of modules to be replaced with FOSSV versions.
    """

    model_type = "fossv"

    def __init__(
        self,
        base_model_name_or_path: Optional[str] = None,
        rank: int = 16,
        alpha: Optional[float] = None,
        dropout: float = 0.0,
        target_modules: List[str] = ["query", "key", "value"],
         mod: str = "small",  # 默认使用原逻辑（后r个奇异值）
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_model_name_or_path = base_model_name_or_path
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.target_modules = target_modules
        self.mod=mod


