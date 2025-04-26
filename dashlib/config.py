from typing import List, Optional
from dataclasses import dataclass, field
from transformers import PretrainedConfig, TrainingArguments

class LoRADashConfig(PretrainedConfig):
    model_type = "lora_dash"
    
    def __init__(
        self,
        base_model_name_or_path: str = None,
        r: int = 16,
        alpha: float=1.0,
        dropout: float = 0.0,
        target_modules: List[str] = ["query", "key", "value"],

        **kwargs
    ):
        super().__init__(**kwargs)
        self.base_model_name_or_path = base_model_name_or_path
        self.r = r
        self.alpha = alpha
        self.dropout = dropout
        self.target_modules = target_modules


@dataclass
class LoRADashTrainingArguments(TrainingArguments):
    def __post_init__(self):
        super().__post_init__()
        self.remove_unused_columns = False