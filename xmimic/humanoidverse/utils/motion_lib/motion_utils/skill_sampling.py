"""Assign environment slots to skills, then sample motions within each skill."""

from collections.abc import Mapping
from pathlib import Path

import joblib
import numpy as np
import torch
from loguru import logger


def sample_motions_by_skill(motions, weights, num_envs, device="cpu"):
    """Return motion indices with largest-remainder skill quotas.

    Weights are relative (e.g. 7:2:1); omitted skills have zero weight. Each
    motion must have one constant one-hot skill_label of shape (T, K). Sampling
    within a skill is uniform over clips, independent of frame/clip counts.
    Torch's seeded RNG controls both motion selection and environment shuffling.
    """
    if num_envs <= 0:
        raise ValueError("skill_sampling requires num_envs > 0")
    if not isinstance(weights, Mapping) or not weights:
        raise ValueError("skill_sampling.weights must be a nonempty skill-ID to weight mapping")
    parsed_weights = {}
    for key, value in weights.items():
        if isinstance(key, bool) or not str(key).isdigit():
            raise ValueError(f"skill_sampling: invalid skill ID {key!r}; use a nonnegative integer")
        skill = int(key)
        weight = float(value)
        if skill in parsed_weights or not np.isfinite(weight) or weight < 0:
            raise ValueError(f"skill_sampling: duplicate skill ID or invalid weight for skill {key!r}")
        parsed_weights[skill] = weight
    active_skills = sorted(skill for skill, weight in parsed_weights.items() if weight > 0)
    if not active_skills:
        raise ValueError("skill_sampling.weights must contain at least one positive weight")

    groups = {}
    label_width = None
    for index, motion in enumerate(motions):
        # Directory-mode motion files use the same filename key as the HOI loader.
        if isinstance(motion, (str, Path)):
            path = Path(motion)
            motion = joblib.load(path)[path.name.split(".")[0]]
        if not isinstance(motion, Mapping) or "skill_label" not in motion:
            raise ValueError(f"Motion {index}: skill_sampling requires skill_label")
        label = np.asarray(motion["skill_label"])
        if (label.ndim != 2 or min(label.shape) == 0
                or label.shape[0] != len(motion["root_trans_offset"])):
            raise ValueError(f"Motion {index}: skill_label must have shape (T, K) matching the motion")
        if (not np.all((label == 0) | (label == 1))
                or label[0].sum() != 1 or not np.all(label == label[0])):
            raise ValueError(f"Motion {index}: skill_label must be a constant one-hot vector across frames")
        if label_width is not None and label.shape[1] != label_width:
            raise ValueError(f"Motion {index}: inconsistent skill_label width")
        label_width = label.shape[1]
        skill = int(label[0].argmax())
        groups.setdefault(skill, []).append(index)
    missing = set(active_skills) - groups.keys()
    if missing:
        raise ValueError(f"skill_sampling: positive weights have no matching motions: {sorted(missing)}")

    relative = np.asarray([parsed_weights[skill] for skill in active_skills], dtype=np.float64)
    relative /= relative.max()
    probabilities = relative / relative.sum()
    expected = probabilities * num_envs
    counts = np.floor(expected).astype(np.int64)
    remainder_order = np.argsort(-(expected - counts), kind="stable")
    counts[remainder_order[:num_envs - int(counts.sum())]] += 1

    sampled = []
    for skill, count, probability in zip(active_skills, counts, probabilities):
        logger.info(f"[Skill sampling] skill={skill}, motions={len(groups[skill])}, "
                    f"envs={count}/{num_envs}, target={probability:.2%}, actual={count / num_envs:.2%}")
        if count == 0:
            logger.warning(f"[Skill sampling] skill={skill} received zero environments after rounding; "
                           "increase num_envs or its weight to include it.")
            continue
        candidates = torch.tensor(groups[skill], device=device, dtype=torch.long)
        sampled.append(candidates[torch.randint(len(candidates), (int(count),), device=device)])
    indices = torch.cat(sampled)
    return indices[torch.randperm(num_envs, device=device)]
