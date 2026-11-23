"""CPU thread and tensor layout settings shared by the trainers."""

import argparse
import os

import torch


def thread_count(value: str) -> int:
    count = int(value)
    if count < 0:
        raise argparse.ArgumentTypeError("use 0 for automatic or a positive thread count")
    return count


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def configure_cpu(requested: int, automatic: int) -> int:
    available = getattr(os, "process_cpu_count", os.cpu_count)() or 1
    chosen = min(requested or automatic, available)
    torch.set_num_threads(chosen)
    return chosen


def training_layout(
    device: torch.device,
    size: int,
    batch_size: int,
    threads: int,
) -> tuple[torch.memory_format, str]:
    if device.type != "cpu":
        return torch.contiguous_format, str(device)
    # Large batches benefit from several workers. Small grids spend proportionally
    # more time synchronizing threads.
    workers = configure_cpu(threads, automatic=4 if size * size * batch_size >= 2048 else 1)
    return torch.channels_last, f"CPU ({workers} threads)"


def training_device(selection: str) -> torch.device:
    if selection == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if selection == "cuda":
        raise ValueError("CUDA is unavailable. Choose auto or CPU.")
    return torch.device("cpu")
