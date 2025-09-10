from dataclasses import dataclass
from typing import Dict, Sequence

import omegaconf
import torch
import transformers


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = [instance["input_ids"] for instance in instances]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        input_ids = input_ids[:, : self.tokenizer.model_max_length]

        batch = {
            "input_ids": input_ids,
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
            "o1_training": True,
        }

        if "labels" in instances[0]:
            labels = [instance["labels"] for instance in instances]
            labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
            labels = labels[:, : self.tokenizer.model_max_length]

            batch["labels"] = labels

        return batch


def webdataset_prompt_collate_fn(batch, prompt_key="caption"):
    prompts = []
    if isinstance(prompt_key, omegaconf.listconfig.ListConfig):
        for key in prompt_key:
            prompts.extend([sample["json"][key] for sample in batch])
    else:
        raise ValueError(f"prompt_key must be a list, got {type(prompt_key)}")
    return {"prompt": prompts}


def json_prompt_collate_fn(batch):
    prompts = [sample["prompt"] for sample in batch]
    # replace the "The image shows" with nothing
    prompts = [prompt.replace("The image shows ", "") for prompt in prompts]
    return {"prompt": prompts}
