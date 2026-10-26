"""Instance discovery and selection (moved verbatim from agent/sci_rollout.py)."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, List, Sequence


def scan_instance_files(data_dir: str, patterns: Iterable[str]) -> List[Path]:
    root = Path(data_dir)
    if not root.exists():
        if "isntance" in str(root):
            raise FileNotFoundError(
                f"Training data directory '{root}' does not exist. "
                "Use 'data/instance/competition' instead of 'data/isntance/competition'."
            )
        raise FileNotFoundError(f"Training data directory '{root}' does not exist.")

    files = []
    for pattern in patterns:
        files.extend(path for path in root.rglob(pattern) if path.is_file())
    files = sorted({
        path.resolve()
        for path in files
        if not path.name.startswith("._") and path.name != "Introduction.txt"
    })
    if not files:
        raise FileNotFoundError(f"No training instance files matched {list(patterns)} under '{root}'.")
    return files


def resolve_instance_files(data_dir: str, patterns: Iterable[str],
                           instance_names: Sequence[str] | None = None) -> List[Path]:
    all_files = scan_instance_files(data_dir, ["*.txt"] if instance_names else patterns)
    if not instance_names:
        return all_files

    by_name = {path.name: path for path in all_files}
    selected = []
    missing = []
    for name in instance_names:
        key = Path(name).name
        if key in by_name:
            selected.append(by_name[key])
        else:
            missing.append(name)
    if missing:
        available = ", ".join(sorted(by_name.keys()))
        raise FileNotFoundError(
            f"Training instances not found under '{data_dir}': {missing}. "
            f"Available instances: {available}"
        )
    return selected


def select_instance(instance_files: List[Path], strategy: str, index: int) -> tuple[int, Path]:
    if not instance_files:
        raise ValueError("No training instances are available.")
    strategy = (strategy or "fixed").lower()
    if strategy == "random":
        selected_index = random.randrange(len(instance_files))
        return selected_index, instance_files[selected_index]
    if strategy in {"fixed", "round_robin"}:
        selected_index = index % len(instance_files)
        return selected_index, instance_files[selected_index]
    raise ValueError("instance_selection must be one of: fixed, round_robin, random")
