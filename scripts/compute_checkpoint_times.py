#!/usr/bin/env python3
"""
Compute approximate training time from checkpoint timestamps and saved epoch info.

Usage examples:

uv run scripts/compute_checkpoint_times.py --checkpoint_dir checkpoints_cnn
uv run scripts/compute_checkpoint_times.py --checkpoint_dir checkpoints_transformer
"""

import argparse
import sys
from pathlib import Path
import time

# Add src/tp1 to path for torch imports if needed (not required here but for uniformity)
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src" / "tp1"))

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Compute checkpoint time statistics")
    parser.add_argument(
        "--checkpoint_dir", required=True, help="Checkpoint folder to inspect"
    )
    return parser.parse_args()


def inspect_checkpoints(checkpoint_dir: str):
    path = Path(checkpoint_dir)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint dir not found: {checkpoint_dir}")

    files = sorted(path.glob("*.pt"))
    if not files:
        print(f"No checkpoint files in {checkpoint_dir}")
        return None

    items = []
    for f in files:
        try:
            ckpt = torch.load(str(f), map_location="cpu")
        except Exception as e:
            print(f"Skipping {f}: can't load checkpoint: {e}")
            continue
        mtime = f.stat().st_mtime
        epoch = None
        if isinstance(ckpt, dict):
            epoch = ckpt.get("epoch", None)
            model_type = ckpt.get("model_type", None)
        else:
            model_type = None
        items.append(
            {"file": f.name, "epoch": epoch, "mtime": mtime, "model_type": model_type}
        )

    # Some checkpoints saved as plain model.pt dict will not contain epoch. Use filename parsing fallback
    for it in items:
        if it["epoch"] is None:
            # try to parse from filename checkpoint_epoch_{num}.pt
            name = it["file"]
            if "checkpoint_epoch_" in name:
                try:
                    it["epoch"] = int(
                        name.split("checkpoint_epoch_")[1].split(".pt")[0]
                    )
                except Exception:
                    it["epoch"] = None
            elif name.startswith("checkpoint_epoch_"):
                try:
                    it["epoch"] = int(name[len("checkpoint_epoch_") : -3])
                except Exception:
                    it["epoch"] = None

    # Sort by epoch if available, else by mtime
    if any(it["epoch"] is not None for it in items):
        items = sorted(
            items,
            key=lambda x: (x["epoch"] if x["epoch"] is not None else float("inf")),
        )  # push none to end
    else:
        items = sorted(items, key=lambda x: x["mtime"])

    # Print summary of files
    print(f"Found {len(items)} checkpoint files in {checkpoint_dir}")
    for it in items:
        epoch_str = str(it["epoch"]) if it["epoch"] is not None else "?"
        mtime_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(it["mtime"]))
        print(
            f" {it['file']:30s} epoch={epoch_str:5s} mtime={mtime_str} model_type={it.get('model_type')}"
        )

    # compute duration from earliest to latest
    # choose earliest and latest with epoch not None if possible
    epoch_items = [it for it in items if it["epoch"] is not None]
    if epoch_items:
        start = epoch_items[0]
        end = epoch_items[-1]
        start_epoch = start["epoch"]
        end_epoch = end["epoch"]
        start_time = start["mtime"]
        end_time = end["mtime"]
        duration_sec = end_time - start_time
        if end_epoch == start_epoch:
            avg_per_epoch = None
        else:
            avg_per_epoch = duration_sec / (end_epoch - start_epoch)
        estimated_total_sec = (
            avg_per_epoch * end_epoch if avg_per_epoch is not None else duration_sec
        )
        print("\nEpoch-based summary:")
        print(
            f" start_epoch={start_epoch}, end_epoch={end_epoch}, duration={duration_sec:.2f}s"
        )
        if avg_per_epoch is not None:
            print(f" avg time per epoch = {avg_per_epoch:.2f}s")
            print(
                f" estimated time to train from epoch 0 to {end_epoch} = {estimated_total_sec:.2f}s"
            )
    else:
        # fallback to mtime-based summary
        start_time = items[0]["mtime"]
        end_time = items[-1]["mtime"]
        duration_sec = end_time - start_time
        print("\nmtime-based summary:")
        print(
            f" start_time={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}"
        )
        print(
            f" end_time=  {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}"
        )
        print(f" duration={duration_sec:.2f}s")
        # since we don't know epoch count, best we can do is average per file saved
        avg_per_file = duration_sec / max(1, (len(items) - 1))
        print(f" avg time per (file) = {avg_per_file:.2f}s")

    return items


if __name__ == "__main__":
    args = parse_args()
    inspect_checkpoints(args.checkpoint_dir)
