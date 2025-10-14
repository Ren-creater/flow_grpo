import os
from dataclasses import dataclass, field
from typing import Optional

from trl.trainer.rloo_config import RLOOConfig


@dataclass
class ConfigPathArguments:
    tokenizer_config: Optional[str] = field(default=None, metadata={"help": "config path of tokenizer"})
    model_config: Optional[str] = field(default=None, metadata={"help": "config path of model"})
    reward_model_config: Optional[str] = field(default=None, metadata={"help": "config path of reward model"})
    train_dataset: Optional[str] = field(default=None, metadata={"help": "config path of training dataset"})
    eval_dataset: Optional[str] = field(default=None, metadata={"help": "config path of evaluation dataset"})
    data_collator: Optional[str] = field(default=None, metadata={"help": "config path of data collator"})


@dataclass
class CustomRLOOConfig(RLOOConfig):
    gamma: float = 0.90
    mean_kl: bool = False
    init_alpha: float = 1.5
    init_beta: float = 0.5
    relative: bool = True
    prediction_type: str = "alpha_beta"
    max_inference_steps: int = 50#28
    evaluation_strategy: str = "steps"
    # Whether to run the in-training evaluation passes (calls to _run_eval_pass()).
    # Default True to preserve existing behavior; set to False to completely disable
    # periodic evaluation and the associated callbacks (wandb image logging, etc.).
    enable_eval: bool = True
