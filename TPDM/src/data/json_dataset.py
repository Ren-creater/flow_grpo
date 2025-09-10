from torch.utils.data import Dataset
from typing import List, Dict
import json
import os

class JsonDataset(Dataset):
    def __init__(self, data_path, tokenizer, add_generation_prompt=False):
        '''Reads a json file or a directory of json files and loads the data into memory.
        Args:
            data_path (str): Path to a json file or a directory of json files.
            tokenizer: A tokenizer object from the transformers library.
            add_generation_prompt (bool): If True, the generation prompt is added to the input.
        '''
        self.tokenizer = tokenizer
        self.add_generation_prompt = add_generation_prompt

        if isinstance(data_path, str):
            if os.path.isdir(data_path):
                # find all json files in the directory
                data_files = []
                for file in os.listdir(data_path):
                    if file.endswith(".json") or file.endswith(".jsonl"):
                        data_files.append(os.path.join(data_path, file))
            else:
                data_files = [data_path]

        self.samples = []
        for data_file in data_files:
            self.samples.extend(self._load_json(data_file))

    def _load_json(self, data_file: str) -> List[Dict]:
        # if the file is a jsonl file
        if data_file.endswith(".jsonl"):
            with open(data_file, "r") as f:
                return [json.loads(line) for line in f]
        # if the file is a json file
        elif data_file.endswith(".json"):
            with open(data_file, "r") as f:
                return json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {data_file}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.tokenizer.apply_chat_template(self.samples[idx], return_tensors="pt", padding=True, return_dict=True, add_generation_prompt=self.add_generation_prompt)
        return {
            "input_ids": item["input_ids"][0],
            "attention_mask": item["attention_mask"][0],
            # "length": torch.as_tensor(item["input_ids"][0].shape[0]),
        }
