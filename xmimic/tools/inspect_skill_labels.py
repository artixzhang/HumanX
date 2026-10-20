#!/usr/bin/env python3
"""Print skill IDs and their motion names from a HumanX PKL (read-only).

Examples from the repository root:
  python xmimic/tools/inspect_skill_labels.py motions.pkl
  python xmimic/tools/inspect_skill_labels.py motions.pkl --limit 0
  python xmimic/tools/inspect_skill_labels.py motions.pkl --skill 0 --limit 0

Skill IDs are zero-based one-hot column indices, as used in skill_sampling.yaml.
Names are the PKL's motion keys, or motion_name/index for a single motion/batch.
Requires only joblib and numpy; no simulator or torch import is needed.
"""

import argparse
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path

import joblib
import numpy as np


def iter_motions(payload, fallback_name):
    if isinstance(payload, Mapping):
        if any(key in payload for key in ("root_pos", "root_trans_offset", "skill_label")):
            yield str(payload.get("motion_name", fallback_name)), payload
        else:
            yield from ((str(name), motion) for name, motion in payload.items())
    elif isinstance(payload, (list, tuple)):
        for index, motion in enumerate(payload):
            name = motion.get("motion_name") if isinstance(motion, Mapping) else None
            yield str(name if name is not None else f"{fallback_name}__{index:04d}"), motion
    else:
        raise ValueError("Expected a named motion dictionary, a single motion, or a list/tuple of motions")


def inspect_labels(payload, fallback_name="motion"):
    groups = defaultdict(list)
    widths = set()
    issues = []
    count = 0
    for name, motion in iter_motions(payload, fallback_name):
        count += 1
        try:
            if not isinstance(motion, Mapping):
                raise ValueError("entry is not a motion dictionary")
            if "skill_label" not in motion:
                raise ValueError("missing skill_label")
            label = np.asarray(motion["skill_label"])
            if label.ndim != 2 or min(label.shape) == 0:
                raise ValueError(f"expected nonempty skill_label (T, K), got {label.shape}")
            widths.add(label.shape[1])
            for key in ("root_trans_offset", "root_pos"):
                if key in motion and len(motion[key]) != label.shape[0]:
                    raise ValueError(f"skill_label frame count differs from {key}")
            if not np.all(np.isfinite(label)):
                raise ValueError("skill_label contains NaN/Inf")
            if not label.any():
                raise ValueError("all-zero skill_label: no assigned skill")
            if not np.all((label == 0) | (label == 1)) or not np.all(label.sum(axis=1) == 1):
                raise ValueError("skill_label is not one-hot in every frame")
            skills = np.unique(label.argmax(axis=1))
            if len(skills) != 1:
                raise ValueError(f"skill changes across frames: {skills.tolist()}")
            groups[int(skills[0])].append(name)
        except (TypeError, ValueError) as exc:
            issues.append((name, str(exc)))
    if count == 0:
        raise ValueError("PKL contains no motions")
    return dict(groups), widths, issues, count


def nonnegative_int(value):
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("filepath", type=Path, help="Motion PKL to inspect")
    parser.add_argument("--limit", type=nonnegative_int, default=10,
                        help="Maximum names per skill / issues to show (default: 10; 0: all)")
    parser.add_argument("--skill", type=nonnegative_int, help="Show only this skill's motion names")
    args = parser.parse_args()
    path = args.filepath.expanduser()
    try:
        groups, widths, issues, count = inspect_labels(joblib.load(path), path.stem)
    except (OSError, TypeError, ValueError, EOFError) as exc:
        parser.error(str(exc))

    print(f"File: {path}")
    print(f"Motions: {count} | Assigned skills: {len(groups)} | Label widths: {sorted(widths)}")
    print("Skill IDs below are zero-based one-hot column indices.")
    if len(widths) > 1:
        print("WARNING: mixed label widths; normalize them before training.")
    for skill, names in sorted(groups.items()):
        if args.skill is not None and args.skill != skill:
            continue
        print(f"\nskill {skill}: {len(names)} motion(s)")
        shown = names[:args.limit] if args.limit else names
        for name in shown:
            print(f"  {name}")
        if len(shown) < len(names):
            print(f"  ... {len(names) - len(shown)} more (use --limit 0 to show all)")
    if args.skill is not None and args.skill not in groups:
        print(f"\nNo motions have a valid constant skill {args.skill}.")
    if issues:
        print(f"\nUnassigned/invalid: {len(issues)} motion(s)")
        shown = issues[:args.limit] if args.limit else issues
        for name, reason in shown:
            print(f"  {name}: {reason}")
        if len(shown) < len(issues):
            print(f"  ... {len(issues) - len(shown)} more (use --limit 0 to show all)")


if __name__ == "__main__":
    main()
