"""Checkpoint helpers (moved verbatim from agent/train.py)."""
from __future__ import annotations

from pathlib import Path

import torch

from agent.ppo import PPO


def save_checkpoint(path: Path, model: PPO, iteration: int, best_fulfillment_rate: float,
                    train_paras: dict, config: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "iteration": int(iteration),
        "best_fulfillment_rate": float(best_fulfillment_rate),
        "policy": model.policy.state_dict(),
        "policy_old": model.policy_old.state_dict(),
        "optimizer": model.optimizer.state_dict(),
        "train_paras": train_paras,
        "config": config,
    }, path)


def load_checkpoint(path: Path, model: PPO, device: torch.device) -> tuple[int, float]:
    checkpoint = torch.load(path, map_location=device)
    model.policy.load_state_dict(checkpoint["policy"])
    model.policy_old.load_state_dict(checkpoint.get("policy_old", checkpoint["policy"]))
    if "optimizer" in checkpoint:
        model.optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint.get("iteration", -1)) + 1, float(checkpoint.get("best_fulfillment_rate", float("-inf")))
