from torch.utils.data import Dataset
from typing import List, Optional
import os


class PlainTextDataset(Dataset):
    """Simple dataset that reads one prompt per line from a text file.

    Returns dicts with key 'prompt' (string). Strips whitespace and ignores empty lines.
    """

    def __init__(self, data_path: str, max_lines: Optional[int] = None, strip_after_comma: bool = False):
        data_path = os.path.expanduser(data_path)
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"data_path not found: {data_path}")
        self.data_path = data_path
        self.strip_after_comma = strip_after_comma

        with open(data_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = [line.rstrip() for line in f]

        # remove empty lines
        lines = [l for l in lines if l.strip()]
        if max_lines is not None:
            lines = lines[:max_lines]

        # optionally strip trailing commas and whitespace
        processed: List[str] = []
        for l in lines:
            if self.strip_after_comma:
                # keep only the text before the first comma (useful if file is prompt,metadata)
                l = l.split(",")[0]
            l = l.strip()
            if l:
                processed.append(l)

        self.samples = processed

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return {"prompt": self.samples[idx]}


if __name__ == "__main__":
    ds = PlainTextDataset("/gpfs/home/zr523/flow_grpo/dataset/pickscore/train.txt")
    print(len(ds))
    print(ds[0])
