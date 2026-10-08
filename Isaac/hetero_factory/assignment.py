"""Which task each env runs (RULES.md section 4).

* Hetero: every rollout holds all tasks. Envs are split as evenly as possible; when ``num_envs`` is not a
  multiple of the task count, the leftover envs change task every episode, so that over every
  ``num_tasks`` episodes each task gets the same number of env-episodes. With 3 envs, one env per task.
* Homo: every env runs the same task. When to switch tasks is decided by the training loop (still an open
  question for the team: per episode, or every N updates); this module only builds the assignment.

Pure torch, no Isaac Lab: the env calls these at reset time.
"""

from __future__ import annotations

import torch


def hetero_assignment(
    num_envs: int, num_tasks: int, episode_idx: int = 0, layout: str = "interleaved", device: str = "cpu"
) -> torch.Tensor:
    """Task id per env for a heterogeneous rollout.

    ``layout`` decides where the tasks sit in env order: ``"interleaved"`` puts env ``i`` on task
    ``i % num_tasks``; ``"blocked"`` gives each task one contiguous block of envs. Both give the same counts;
    whether the order changes GPU speed is something to measure.
    """
    if num_envs < num_tasks:
        raise ValueError(f"need at least one env per task: num_envs={num_envs}, num_tasks={num_tasks}")
    base, extra = divmod(num_envs, num_tasks)
    if layout == "interleaved":
        ids = torch.arange(num_envs, device=device) % num_tasks
        # the leftover envs are the last `extra`; rotate them by the episode index
        if extra:
            tail = torch.arange(num_envs - extra, num_envs, device=device)
            ids[tail] = (torch.arange(extra, device=device) + episode_idx) % num_tasks
    elif layout == "blocked":
        ids = torch.arange(num_tasks, device=device).repeat_interleave(base)
        if extra:
            ids = torch.cat([ids, (torch.arange(extra, device=device) + episode_idx) % num_tasks])
    else:
        raise ValueError(f"unknown layout {layout!r}")
    return ids.long()


def counts_assignment(counts, num_envs: int, device: str = "cpu") -> torch.Tensor:
    """Analysis only (2026-10-07 meeting, TODO2/3): a fixed number of envs per task, in contiguous blocks (task 0
    first). Counts may be uneven or zero, e.g. more envs for a cheap task, or only two of the three tasks."""
    counts = [int(c) for c in counts]
    if sum(counts) != num_envs or min(counts) < 0:
        raise ValueError(f"task counts {counts} must be >= 0 and sum to num_envs={num_envs}")
    return torch.arange(len(counts), device=device).repeat_interleave(torch.tensor(counts, device=device)).long()


def homo_assignment(num_envs: int, task_id: int, device: str = "cpu") -> torch.Tensor:
    """Every env on the same task."""
    return torch.full((num_envs,), int(task_id), dtype=torch.long, device=device)


def task_counts(task_ids: torch.Tensor, num_tasks: int) -> list[int]:
    return torch.bincount(task_ids, minlength=num_tasks).tolist()
