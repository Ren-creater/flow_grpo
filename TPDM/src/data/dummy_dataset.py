import torch
from torch.utils.data import Dataset


sample1 = [{
    "role": "user",
    "content": "Hello, how are you?"
}, {
    "role": "assistant",
    "content": "I'm fine, thank you. How can I help you today?"
}]
sample2 = [{
    "role": "user",
    "content": "Good morning!"
}, {
    "role": "assistant",
    "content": "Good morning!"
    }]


class DummyDataset(Dataset):
    def __init__(self, tokenizer, add_generation_prompt=False):
        self.samples = [sample1, sample2, sample1, sample2, sample1, sample2, sample1, sample2, sample1, sample2]
        self.tokenizer = tokenizer
        self.add_generation_prompt = add_generation_prompt
        if self.add_generation_prompt:
            self.samples = [sam[:-1] for sam in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.tokenizer.apply_chat_template(self.samples[idx], return_tensors="pt", padding=True, return_dict=True, add_generation_prompt=self.add_generation_prompt)
        return {
            "input_ids": item["input_ids"][0],
            "attention_mask": item["attention_mask"][0],
            # "length": torch.as_tensor(item["input_ids"][0].shape[0]),
        }


if __name__ == "__main__":
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("/mnt/sda/models/Meta-Llama-3.1-8B-Instruct/", pad_token="<|reserved_special_token_247|>")
    dataset = DummyDataset(tokenizer, add_generation_prompt=True)
    print(dataset[0])

    from torch.utils.data import DataLoader
    from transformers import DataCollatorWithPadding
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=DataCollatorWithPadding(tokenizer),
        drop_last=True,  # needed; otherwise the last batch will be of ragged shape
        num_workers=0,
    )
    def get_item():
        while True:
            yield from dataloader

    item = get_item()
    print(next(item))