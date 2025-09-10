import glob
import logging
import os
import random
from functools import partial

import omegaconf
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

import datasets

logger = logging.getLogger(__name__)

def get_hf_webdataset(tar_path, shuffle=True, **kwarg):
    if "*tar" in tar_path:
        tar_files = glob.glob(tar_path)
    else:
        tar_files = glob.glob(os.path.join(tar_path, "*.tar"))
    if shuffle:
        generator = random.Random(42)
        generator.shuffle(tar_files)
    dataset = datasets.load_dataset(
        "webdataset",
        data_files=tar_files,
        split="train",
        **kwarg,
    )
    if shuffle:
        dataset = dataset.shuffle(seed=42, buffer_size=10_000)
    return dataset


def get_hf_json(json_path, shuffle=True, **kwarg):
    # judge json_path if iterable
    if isinstance(json_path, str):
        if "*json" in json_path or "*jsonl" in json_path:
            json_files = glob.glob(json_path)
        else:
            json_files = glob.glob(os.path.join(json_path, "*.json"))
    elif isinstance(json_path, list) or isinstance(json_path, omegaconf.listconfig.ListConfig):
        json_files = []
        for path in json_path:
            if "*json" in path or "*jsonl" in path:
                json_files.extend(glob.glob(path))
            else:
                json_files.extend(glob.glob(os.path.join(path, "*.json")))
    else:
        raise ValueError(f"json_path must be str or list, but got {type(json_path)}")

    if shuffle:
        generator = random.Random(42)
        generator.shuffle(json_files)

    logger.info(f"Found {len(json_files)} json files")

    dataset = datasets.load_dataset(
        "json",
        data_files=json_files,
        split="train",
        **kwarg,
    )
    if shuffle:
        dataset = dataset.shuffle(seed=42)
    return dataset


if __name__ == "__main__":
    dataset = get_hf_webdataset("/qiguojun/home/Dataset/laion-art-recap-1024-328k/*tar")
    for sample in tqdm(dataset):
        pass
