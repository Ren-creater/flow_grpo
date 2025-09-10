import logging
import os
import pathlib

import hydra
import torch
import transformers
from omegaconf import OmegaConf

# WARNING: the args for trainer are from the official Arguments, do not refer to that in our config.py
from src.train.callbacks import DiffusionWandbCallback
from src.train.config import ConfigPathArguments, CustomRLOOConfig
from src.train.rloo_trainer import CommonRLOOTrainer
from src.train.train_utilis import setup_debug


# rank = os.environ.get("RANK")
# setup_debug(int(rank))


log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)

logger = logging.getLogger(__name__)


def train(cfg, training_args):
    model = hydra.utils.instantiate(
        OmegaConf.load(cfg.model_config),
        init_alpha=training_args.init_alpha,
        init_beta=training_args.init_beta,
        relative=training_args.relative,
        prediction_type=training_args.prediction_type,
        fsdp=training_args.fsdp,
        max_inference_steps=training_args.max_inference_steps,
    )
    logger.info(f"model loaded from {cfg.model_config}")
    reward_model = hydra.utils.instantiate(OmegaConf.load(cfg.reward_model_config)).eval()
    logger.info(f"reward model loaded from {cfg.reward_model_config}")
    train_dataset = hydra.utils.instantiate(OmegaConf.load(cfg.train_dataset))
    logger.info(f"train dataset loaded from {cfg.train_dataset}")
    data_collator = hydra.utils.instantiate(OmegaConf.load(cfg.data_collator))
    logger.info(f"data collator loaded from {cfg.data_collator}")

    # TODO: someone may prefer save training_args in a file
    trainer = CommonRLOOTrainer(
        config=training_args,
        policy=model,
        reward_model=reward_model,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=train_dataset,
    )

    if "wandb" in training_args.report_to:
        wandb_callback = DiffusionWandbCallback(trainer=trainer)
        logger.info("wandb callback added")
        trainer.add_callback(wandb_callback)

    if (
        list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")) and args.resume_from_checkpoint is not None
    ) or os.path.isdir(args.resume_from_checkpoint):
        # jugde whether resume_from_checkpoint is a path
        if os.path.isdir(args.resume_from_checkpoint):
            trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        else:
            trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()



if __name__ == "__main__":
    parser = transformers.HfArgumentParser((ConfigPathArguments, CustomRLOOConfig))
    cfg, args = parser.parse_args_into_dataclasses()

    train(cfg, args)
