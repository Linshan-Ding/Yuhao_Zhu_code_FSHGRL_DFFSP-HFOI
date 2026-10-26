"""How the *current* tree is exercised by the harness.

This is the only harness file that changes while the layout migrates: it must
know where the instance lives and how the trainer is started.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import yaml

from .common import INSTANCE_NAME, MAX_ITERATIONS, ROOT, SEED, WORK_DIR, golden_train_overrides, run_subprocess


def instance_path() -> Path:
    return ROOT / "data" / "instance" / "competition" / INSTANCE_NAME


def overlay_for(model_type: str, experiment_name: str) -> dict:
    """The golden train_paras overrides expressed in the YAML layout."""
    o = golden_train_overrides(model_type)
    return {
        "instance": {"train_instances": o["train_instances"], "instance_selection": o["instance_selection"]},
        "env": {"fluid_mode": o["fluid_mode"]},
        "algo": {
            "model_type": o["model_type"],
            "train": {key: o[key] for key in ("num_envs", "num_workers", "minibatch_size", "max_minibatch_tokens",
                                               "max_iterations", "resume", "rollout_device", "update_device")}
            | {"experiment_name": experiment_name},
            "viz": {"viz": o["viz"]},
        },
    }


def run_candidate(base_config: dict, model_type: str, log_path: Path) -> Path:
    """Run the current tree's trainer for the reference configuration.

    Returns the directory holding train_metrics.csv and the checkpoints.
    """
    experiment_name = f"eq_{model_type}"
    shutil.rmtree(ROOT / "result" / experiment_name, ignore_errors=True)  # the trainer appends to an existing CSV
    overlay_path = WORK_DIR / f"overlay_{model_type}.yaml"
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    overlay_path.write_text(yaml.safe_dump(overlay_for(model_type, experiment_name), sort_keys=False), encoding="utf-8")
    cmd = [
        sys.executable, "train.py",
        "--config", "configs/instance.yaml", "configs/env.yaml", "configs/algo.yaml", str(overlay_path),
        "--seed", str(SEED),
        "--max-iterations", str(MAX_ITERATIONS),
        "--rollout-device", "cpu",
        "--update-device", "cpu",
        "--resume", "none",
        "--instances", INSTANCE_NAME,
        "--experiment-name", experiment_name,
    ]
    run_subprocess(cmd, ROOT, log_path)
    return ROOT / "result" / experiment_name
