"""How the *current* tree is exercised by the harness.

This is the only harness file that changes while the layout migrates: it must
know where the instance lives and how the trainer is started.
"""
from __future__ import annotations

from pathlib import Path

from .common import INSTANCE_NAME, ROOT, run_legacy_train


def instance_path() -> Path:
    return ROOT / "data" / "instance" / "competition" / INSTANCE_NAME


def run_candidate(base_config: dict, model_type: str, log_path: Path) -> Path:
    """Run the current tree's trainer for the reference configuration.

    Returns the directory holding train_metrics.csv and the checkpoints.
    """
    return run_legacy_train(ROOT, base_config, model_type, f"eq_{model_type}", log_path)
