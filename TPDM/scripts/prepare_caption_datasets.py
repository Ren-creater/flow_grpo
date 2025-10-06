#!/usr/bin/env python3
"""Download caption-only text corpora from public datasets.

This utility pulls caption annotations from three large-scale datasets:

* MS-COCO captions (2017 train/val)
* CaptionEmporium/coyo-hd-11m-llavanext (Hugging Face hub)
* laion/relaion-art (Hugging Face hub)

Only the textual captions are retained; image assets are not downloaded. The
results are written to newline-delimited text files that are compatible with
`PlainTextDataset` configs used by the TPDM training scripts. A single combined
file aggregating all captions can also be produced for convenience.

Example usage:

```
python TPDM/scripts/prepare_caption_datasets.py \
    --output-root dataset/caption_exports \
    --max-coco 0 --max-coyo 1000000 --max-relaion 0
```

The example above will stream the full COYO split, skip MS-COCO and ReLAION, and
write the first one million COYO captions to the output directory.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Iterable, Iterator, List, Optional
from zipfile import ZipFile

import requests

try:
    from datasets import IterableDataset, load_dataset  # type: ignore
except ImportError:  # pragma: no cover - handled at runtime
    load_dataset = None  # type: ignore
    IterableDataset = None  # type: ignore


COCO_ANNOTATION_ZIP = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
COCO_CAPTION_FILES = {
    "train": "annotations/captions_train2017.json",
    "val": "annotations/captions_val2017.json",
}

# Candidate fields to probe for caption text in Hugging Face datasets.
DEFAULT_TEXT_FIELDS = [
    "caption",
    "captions",
    "text",
    "TEXT",
    "prompt",
    "description",
    "message",
]


def sanitize_text(text: str) -> str:
    """Normalize whitespace and strip control characters."""
    text = text.replace("\r", " ").replace("\n", " ")
    text = " ".join(text.split())
    return text.strip()


def iter_json_captions(json_path: Path, limit: Optional[int] = None) -> Iterator[str]:
    """Yield captions from an MS-COCO JSON annotation file."""
    logging.debug("Parsing COCO captions from %s", json_path)
    with json_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    annotations = data.get("annotations", [])
    count = 0
    for ann in annotations:
        caption = ann.get("caption")
        if not isinstance(caption, str):
            continue
        caption = sanitize_text(caption)
        if not caption:
            continue
        yield caption
        count += 1
        if limit and count >= limit:
            break


def download_file(url: str, destination: Path, overwrite: bool = False) -> Path:
    """Download a large file with streaming requests."""
    if destination.exists() and not overwrite:
        logging.info("Using cached file %s", destination)
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    logging.info("Downloading %s -> %s", url, destination)
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("Content-Length", 0))
        bytes_read = 0
        chunk_size = 1 << 20  # 1 MiB
        with destination.open("wb") as out:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                out.write(chunk)
                bytes_read += len(chunk)
                if total:
                    pct = (bytes_read / total) * 100
                    sys.stdout.write(f"\r  {bytes_read/1e6:,.1f} MB / {total/1e6:,.1f} MB ({pct:.1f}%)")
                else:
                    sys.stdout.write(f"\r  {bytes_read/1e6:,.1f} MB downloaded")
                sys.stdout.flush()
    sys.stdout.write("\n")
    return destination


def write_captions(
    captions: Iterable[str],
    output_path: Path,
    combined_handle,
    overwrite: bool,
) -> int:
    """Write captions to disk and append to the combined file if provided."""
    if output_path.exists() and not overwrite:
        logging.info("Skipping existing file %s", output_path)
        with output_path.open("r", encoding="utf-8") as existing:
            return sum(1 for _ in existing)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for caption in captions:
            handle.write(caption + "\n")
            if combined_handle is not None:
                combined_handle.write(caption + "\n")
            count += 1
            if count % 100000 == 0:
                logging.debug("  Wrote %d captions to %s", count, output_path.name)
    logging.info("Saved %d captions to %s", count, output_path)
    return count


def prepare_coco(
    output_root: Path,
    combined_handle,
    overwrite: bool,
    max_examples: Optional[int],
) -> int:
    """Download and export MS-COCO captions."""
    coco_root = output_root / "coco"
    coco_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_zip_path = Path(tmp_dir) / "annotations_trainval2017.zip"
        download_file(COCO_ANNOTATION_ZIP, tmp_zip_path, overwrite=overwrite)
        with ZipFile(tmp_zip_path, "r") as zip_ref:
            json_paths: List[Path] = []
            for split, member in COCO_CAPTION_FILES.items():
                if member not in zip_ref.namelist():
                    logging.warning("Missing %s inside COCO archive", member)
                    continue
                logging.info("Extracting %s", member)
                extracted = zip_ref.extract(member, path=tmp_dir)
                json_paths.append(Path(extracted))

            counts = 0
            remaining = max_examples
            for json_path in json_paths:
                split_name = json_path.stem.replace("captions_", "")
                out_path = coco_root / f"{split_name}.txt"
                if remaining is not None and remaining <= 0:
                    break
                limit = remaining if remaining is not None else None
                captions = iter_json_captions(json_path, limit=limit)
                written = write_captions(captions, out_path, combined_handle, overwrite=overwrite)
                counts += written
                if remaining is not None:
                    remaining -= written
    return counts


def extract_candidate_strings(example: dict, fields: List[str]) -> Iterator[str]:
    """Yield text strings from a dataset example using common field names."""
    for field in fields:
        if field not in example:
            continue
        value = example[field]
        if isinstance(value, str):
            sanitized = sanitize_text(value)
            if sanitized:
                yield sanitized
        elif isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, str):
                    sanitized = sanitize_text(item)
                    if sanitized:
                        yield sanitized
                elif isinstance(item, dict):
                    for maybe_text in extract_candidate_strings(item, fields):
                        yield maybe_text
        elif isinstance(value, dict):
            for maybe_text in extract_candidate_strings(value, fields):
                yield maybe_text


def prepare_hf_dataset(
    dataset_name: str,
    split: str,
    output_root: Path,
    file_stem: str,
    combined_handle,
    overwrite: bool,
    max_examples: Optional[int],
    text_fields: Optional[List[str]] = None,
    hf_token: Optional[str] = None,
    trust_remote_code: bool = False,
) -> int:
    """Stream a Hugging Face dataset and export its captions."""
    if load_dataset is None:
        raise ImportError(
            "The 'datasets' package is required. Install via `pip install datasets` inside the tpdm environment."
        )

    text_fields = text_fields or DEFAULT_TEXT_FIELDS
    out_dir = output_root / dataset_name.replace("/", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{file_stem}.txt"

    if output_path.exists() and not overwrite:
        logging.info("Skipping existing file %s", output_path)
        with output_path.open("r", encoding="utf-8") as existing:
            return sum(1 for _ in existing)

    logging.info("Loading %s[%s] from Hugging Face hub", dataset_name, split)
    # Some dataset builders (ParquetConfig etc.) don't accept unexpected
    # builder config keys like `use_auth_token`. Try the common kw first and
    # if the builder raises a ValueError re-trying with the alternate token
    # argument name used by newer/older `datasets` versions.
    load_kwargs = dict(split=split, streaming=True, trust_remote_code=trust_remote_code)
    if hf_token:
        load_kwargs["use_auth_token"] = hf_token

    try:
        dataset = load_dataset(dataset_name, **load_kwargs)
    except ValueError as e:
        # Example error message: "BuilderConfig ... doesn't have a 'use_auth_token' key."
        msg = str(e)
        if hf_token and "doesn't have a 'use_auth_token' key" in msg:
            # Retry using the alternate `token` kwarg
            load_kwargs.pop("use_auth_token", None)
            load_kwargs["token"] = hf_token
            logging.debug("Retrying load_dataset with 'token' instead of 'use_auth_token'")
            dataset = load_dataset(dataset_name, **load_kwargs)
        else:
            raise

    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for example in dataset:  # type: ignore[assignment]
            found_any = False
            for caption in extract_candidate_strings(example, text_fields):
                handle.write(caption + "\n")
                if combined_handle is not None:
                    combined_handle.write(caption + "\n")
                count += 1
                found_any = True
                if count % 100000 == 0:
                    logging.debug("  Wrote %d captions to %s", count, output_path.name)
                if max_examples and count >= max_examples:
                    break
            if not found_any:
                logging.debug("No caption fields found in example keys=%s", list(example.keys())[:10])
            if max_examples and count >= max_examples:
                break
    logging.info("Saved %d captions to %s", count, output_path)
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download/export caption text corpora")
    default_output = Path(__file__).resolve().parents[2] / "dataset" / "caption_exports"
    parser.add_argument(
        "--output-root",
        type=Path,
        default=default_output,
        help="Where to store exported caption text files (default: %(default)s)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Recreate files even if they already exist")
    parser.add_argument("--skip-coco", action="store_true", help="Skip MS-COCO captions")
    parser.add_argument("--skip-coyo", action="store_true", help="Skip CaptionEmporium/coyo-hd-11m-llavanext")
    parser.add_argument("--skip-relaion", action="store_true", help="Skip laion/relaion-art")
    parser.add_argument("--max-coco", type=int, default=None, help="Optional max captions to export from COCO")
    parser.add_argument("--max-coyo", type=int, default=None, help="Optional max captions to export from COYO")
    parser.add_argument(
        "--max-relaion",
        type=int,
        default=None,
        help="Optional max captions to export from ReLAION",
    )
    parser.add_argument(
        "--combined-name",
        type=str,
        default="all_captions.txt",
        help="Name of the combined caption file inside the output root",
    )
    parser.add_argument(
        "--skip-combined",
        action="store_true",
        help="Do not maintain a combined caption file (per-dataset files still created)",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN"),
        help="Hugging Face token for gated datasets (defaults to HF_TOKEN env var)",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to load_dataset (required for some datasets)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="[%(levelname)s] %(message)s")

    output_root: Path = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    combined_path = output_root / args.combined_name
    combined_handle = None
    if not args.skip_combined:
        combined_handle = combined_path.open("w" if args.overwrite else "a", encoding="utf-8")
        if args.overwrite:
            logging.info("Combined captions will be written to %s", combined_path)

    summary = {}

    try:
        if not args.skip_coco:
            total = prepare_coco(output_root, combined_handle, args.overwrite, args.max_coco)
            summary["coco"] = total
        else:
            logging.info("Skipping COCO export per user request")

        if not args.skip_coyo:
            total = prepare_hf_dataset(
                dataset_name="CaptionEmporium/coyo-hd-11m-llavanext",
                split="train",
                output_root=output_root,
                file_stem="train",
                combined_handle=combined_handle,
                overwrite=args.overwrite,
                max_examples=args.max_coyo,
                text_fields=DEFAULT_TEXT_FIELDS,
                hf_token=args.hf_token,
                trust_remote_code=args.trust_remote_code,
            )
            summary["coyo-hd-11m-llavanext"] = total
        else:
            logging.info("Skipping COYO export per user request")

        if not args.skip_relaion:
            total = prepare_hf_dataset(
                dataset_name="laion/relaion-art",
                split="train",
                output_root=output_root,
                file_stem="train",
                combined_handle=combined_handle,
                overwrite=args.overwrite,
                max_examples=args.max_relaion,
                text_fields=DEFAULT_TEXT_FIELDS,
                hf_token=args.hf_token,
                trust_remote_code=args.trust_remote_code,
            )
            summary["relaion-art"] = total
        else:
            logging.info("Skipping ReLAION export per user request")
    finally:
        if combined_handle is not None:
            combined_handle.close()

    logging.info("Export summary: %s", ", ".join(f"{ds}={count:,}" for ds, count in summary.items()))
    logging.info("Caption text files ready under %s", output_root)


if __name__ == "__main__":
    main()
