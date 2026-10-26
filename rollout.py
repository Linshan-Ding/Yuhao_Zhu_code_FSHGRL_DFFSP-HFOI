"""Synchronous multi-environment episode collection (moved verbatim from agent/sci_rollout.py)."""
from __future__ import annotations

import time
from pathlib import Path

import torch

from agent.buffer import RolloutBuffer, _mark_last_transition_terminal
from data.instance_io import parse_instance_file
from environment.simulator import TrainingSimulator


def collect_vectorized_rollouts(policy, instance_path: Path, train_paras: dict,
                                base_trajectory_id: int = 0) -> RolloutBuffer:
    spec = parse_instance_file(instance_path)
    num_envs = int(train_paras.get("num_envs", train_paras.get("num_workers", 4)))
    buffer = RolloutBuffer(minibatch_size=int(train_paras.get("minibatch_size", 512)))
    buffer.max_minibatch_tokens = int(train_paras.get("max_minibatch_tokens", 0))
    simulators = [TrainingSimulator(spec, train_paras) for _ in range(num_envs)]
    trajectory_ids = [base_trajectory_id + index for index in range(num_envs)]
    episode_start_times = [time.perf_counter() for _ in range(num_envs)]
    finished = [False for _ in range(num_envs)]
    policy_forward_count = 0
    policy_forward_seconds = 0.0
    total_requests = 0
    action_tokens_sum = 0.0
    max_action_tokens = 0.0
    padding_ratio_sum = 0.0

    while not all(finished):
        active_indices = [
            index for index, simulator in enumerate(simulators)
            if not finished[index] and not simulator.done and simulator.has_decision()
        ]
        if not active_indices:
            for index, simulator in enumerate(simulators):
                if not finished[index] and simulator.done:
                    finished[index] = True
                    _mark_last_transition_terminal(buffer, trajectory_ids[index])
                    buffer.fulfillment_rates.append(simulator.fulfillment_rate)
                    buffer.discard_count += simulator.discarded_count
                    buffer.completion_count += simulator.completed_count
                    buffer.fluid_solve_count += simulator.fluid_solve_count
                    buffer.fluid_solve_seconds += simulator.fluid_solve_seconds
                    buffer.fluid_cache_hit_count += simulator.fluid_cache_hit_count
                    buffer.fluid_fallback_count += simulator.fluid_fallback_count
                    buffer.episode_transition_counts.append(simulator.transition_count)
                    buffer.episode_seconds.append(time.perf_counter() - episode_start_times[index])
                    buffer.obs_build_seconds += simulator.obs_build_seconds
                    buffer.simulator_step_seconds += simulator.step_seconds
            continue

        obs_batch = [simulators[index].to_policy_obs() for index in active_indices]
        forward_start = time.perf_counter()
        with torch.no_grad():
            action_indexes, log_probs, values, env_actions = policy.act_batch(obs_batch, sample=True)
        forward_elapsed = time.perf_counter() - forward_start
        stats = getattr(policy, "last_forward_stats", {}) or {}
        batch_size = int(stats.get("batch_size", len(obs_batch)))
        policy_forward_count += 1
        policy_forward_seconds += float(stats.get("forward_seconds", forward_elapsed))
        total_requests += batch_size
        action_tokens_sum += float(stats.get("avg_action_tokens", 0.0)) * batch_size
        max_action_tokens = max(max_action_tokens, float(stats.get("max_action_tokens", 0.0)))
        padding_ratio_sum += float(stats.get("padding_ratio", 0.0)) * batch_size

        for local_index, env_index in enumerate(active_indices):
            simulator = simulators[env_index]
            reward = simulator.step(env_actions[local_index])
            done = simulator.done
            buffer.add(
                obs_batch[local_index],
                int(action_indexes[local_index].detach().cpu().item()),
                env_actions[local_index],
                float(log_probs[local_index].detach().cpu().item()),
                float(values[local_index].detach().cpu().item()),
                reward,
                done,
                trajectory_ids[env_index],
            )

    request_count = max(total_requests, 1)
    buffer.policy_forward_count = policy_forward_count
    buffer.policy_forward_seconds = policy_forward_seconds
    buffer.avg_action_tokens = action_tokens_sum / request_count
    buffer.max_action_tokens = max_action_tokens
    buffer.padding_ratio = padding_ratio_sum / request_count
    buffer.actor_batch_size_mean = total_requests / max(policy_forward_count, 1)
    return buffer


def _episode_record(simulator: TrainingSimulator, total_reward: float, wall_seconds: float) -> dict:
    return {
        "fulfillment_rate": float(simulator.fulfillment_rate),
        "completed": int(simulator.completed_count),
        "discarded": int(simulator.discarded_count),
        "total_orders": int(simulator.spec.total_orders),
        "decision_count": int(simulator.transition_count),
        "schedule_count": int(simulator.schedule_count),
        "total_reward": float(total_reward),
        "wall_seconds": float(wall_seconds),
        "obs_build_seconds": float(simulator.obs_build_seconds),
        "simulator_step_seconds": float(simulator.step_seconds),
        "fluid_solve_count": int(simulator.fluid_solve_count),
        "fluid_solve_seconds": float(simulator.fluid_solve_seconds),
        "fluid_cache_hit_count": int(simulator.fluid_cache_hit_count),
        "fluid_fallback_count": int(simulator.fluid_fallback_count),
    }


def run_episodes(policy, spec, train_paras: dict, episodes: int = 1, sample: bool = False,
                 simulator_factory=TrainingSimulator) -> list:
    """Run ``episodes`` complete episodes of one instance in lockstep and return one record per episode.

    New code (no counterpart in the original repository). It drives the simulators exactly like
    ``collect_vectorized_rollouts`` but keeps no buffer. With ``sample=False`` the policy decodes greedily
    (argmax of both action distributions) and draws **no random numbers**, which is what lets ``train.py``
    evaluate the policy after every update without disturbing the training stream (checked by
    ``tests/equivalence``). ``sample=True`` draws actions from the policy and is for ``eval.py`` only.

    With ``episodes > 1`` the observations of all running episodes share one forward pass; the policy's
    per-batch normalisation (due dates, processing rates) then sees the batch as a whole, so numbers that
    must not depend on the batch composition are produced with ``episodes=1``.
    """
    simulators = [simulator_factory(spec, train_paras) for _ in range(episodes)]
    start_times = [time.perf_counter() for _ in range(episodes)]
    finished = [False for _ in range(episodes)]
    total_rewards = [0.0 for _ in range(episodes)]
    records: list = [None for _ in range(episodes)]
    forward_seconds = 0.0
    forward_count = 0

    while not all(finished):
        active_indices = [
            index for index, simulator in enumerate(simulators)
            if not finished[index] and not simulator.done and simulator.has_decision()
        ]
        if not active_indices:
            for index, simulator in enumerate(simulators):
                if not finished[index] and simulator.done:
                    finished[index] = True
                    records[index] = _episode_record(simulator, total_rewards[index],
                                                     time.perf_counter() - start_times[index])
            continue

        obs_batch = [simulators[index].to_policy_obs() for index in active_indices]
        forward_start = time.perf_counter()
        with torch.no_grad():
            _, _, _, env_actions = policy.act_batch(obs_batch, sample=sample)
        forward_seconds += time.perf_counter() - forward_start
        forward_count += 1
        for local_index, env_index in enumerate(active_indices):
            total_rewards[env_index] += simulators[env_index].step(env_actions[local_index])

    for record in records:
        record["policy_forward_seconds"] = forward_seconds   # shared by the lockstep batch
        record["policy_forward_count"] = forward_count
        record["batch_episodes"] = episodes
    return records
