#!/usr/bin/env python3
"""Convert HumanX G1 HOI PKLs to BB, with one output file per skill.

Examples (from the repository root)::

    python xmimic/tools/humanx_to_bb.py motions.pkl -o motions_bb.pkl
    python xmimic/tools/humanx_to_bb.py motions.pkl -o bb_skills/

Inputs accept a single motion, a list/tuple, or a named motion dictionary.
Each output contains a flat motion dict for a single trajectory, or a list of
motion dicts for multiple trajectories, matching the BB editor/viewer/batch
tools. No outer {motion_name: motion} wrapper is saved. The single-motion editor
needs a flat dict; use BB's manage_batch.py --split for multi-trajectory files.
With one skill, a .pkl output path is used directly. With multiple skills it becomes
<stem>_skill_<ID>.pkl for each original zero-based one-hot column index. A
directory output always contains skill_<ID>.pkl files. Missing/all-zero labels
form an "unlabelled" group, separate from skill 0. Invalid labels are rejected.
If a motion changes skill across frames, its contiguous runs become separate
motions; disjoint runs are never concatenated. Time-leading arrays/lists are
sliced with each run, while known static metadata is retained. stand_frame is
rebased and clamped to the segment. skill_label is removed after grouping.
Even one/two-frame segments are preserved; the existing bb_to_humanx importer
requires at least three frames per motion when converting back to HumanX.

root_trans_offset -> root_pos; obj_rot_quat -> obj_rot; root/object rotations
change XYZW -> WXYZ; dof becomes (T,29) in HumanX joint order with dof_names;
basket_pos -> hoop_pos_w (constant positions become (3,)); obj_freeze ->
freeze_obj. Contact columns are left foot, right foot, left hand, right hand.
HumanX's aggregate hand contact cannot recover separate left/right contacts,
so it is copied to both hand columns. Existing four-column contact is retained.
Missing foot contacts use 0.5, matching bb_to_humanx's unknown value.

Existing push values are preserved independently. push_available, obj_push and
an existing release are normalized to (T,), broadcasting constants and squeezing
(T,1) columns without changing values/dtype. Missing push_available and obj_push
become zero vectors of length T. release
and stand_frame remain independent of contact, freezing, and push labels.
pose_aa/pos_aa, known robot FK caches, obj_rot_euler and finite-difference object
velocities are removed; bb_to_humanx/the HumanX motion loader can rebuild them.
Other fields keep their original keys. No original-data backup is embedded.
FPS, world coordinates and frame order are preserved; no simulator is needed.
Requires numpy, scipy and joblib, like bb_to_humanx.py.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

import joblib
import numpy as np

if __package__:
    from .bb_to_humanx import (
        CONTACT_NAMES, HUMANX_DOF_NAMES, _array, _contact, _integer, _motions,
        _quaternion, _resolve, _sequence,
    )
else:
    from bb_to_humanx import (
        CONTACT_NAMES, HUMANX_DOF_NAMES, _array, _contact, _integer, _motions,
        _quaternion, _resolve, _sequence,
    )


# Explicit keys only: unrelated human-reference data and unknown fields survive.
DERIVED_FIELDS = frozenset({
    "pose_aa", "pos_aa", "normalized_pose_aa", "obj_rot_euler",
    "obj_vel", "obj_ang_vel", "global_translation", "global_rotation_mat",
    "global_rotation", "local_rotation", "global_root_velocity",
    "global_root_angular_velocity", "global_angular_velocity", "global_velocity",
    "dof_pos", "dof_vels", "global_velocity_extend",
    "global_angular_velocity_extend", "global_translation_extend",
    "global_rotation_mat_extend", "global_rotation_extend",
})
STATIC_FIELDS = frozenset({
    "fps", "dof_names", "contact_names", "object_size", "support_top_pos",
    "hoop_pos_cyl", "motion_name", "source_path", "need_support",
})


def _bb_contacts(data: dict, frames: int) -> np.ndarray:
    if "contact" not in data:
        raise ValueError("Missing required field contact; cannot infer it from release")
    raw = _array(data.pop("contact"), "contact")
    names = data.pop("contact_names", None)
    if raw.shape == (frames, 4):
        names = list(CONTACT_NAMES if names is None else names)
        if len(names) != 4 or set(names) != set(CONTACT_NAMES):
            raise ValueError(f"contact_names must contain exactly {CONTACT_NAMES}")
        contacts = raw[:, [names.index(name) for name in CONTACT_NAMES]].copy()
        if np.any((contacts < 0) | (contacts > 1)):
            raise ValueError("contact must contain values in [0, 1]")
        for index, key in enumerate(("lfoot_contact", "rfoot_contact")):
            if key in data:
                foot = _contact(data.pop(key), key, frames)
                if not np.allclose(contacts[:, index:index + 1], foot):
                    raise ValueError(f"{key} conflicts with four-column contact")
        return contacts
    hand = _contact(raw, "contact", frames)
    feet = [
        _contact(data.pop(key), key, frames) if key in data
        else np.full((frames, 1), 0.5, dtype=np.float32)
        for key in ("lfoot_contact", "rfoot_contact")
    ]
    return np.concatenate((*feet, hand, hand), axis=1)


def _frame_label(value, key: str, frames: int) -> np.ndarray:
    """BB readers require a vector; keep labels' values and original dtype."""
    values = np.asarray(value)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{key} contains non-finite values")
    if values.shape in ((), (1,), (1, 1)):
        return np.full(frames, values.reshape(-1)[0], dtype=values.dtype)
    if values.shape in ((frames,), (frames, 1)):
        return values.reshape(frames).copy()
    raise ValueError(f"{key} has shape {values.shape}; expected a scalar, ({frames},) or ({frames},1)")


def convert_motion(source: dict, *, quat_order: str = "xyzw") -> dict:
    """Convert one motion without mutating its source dictionary or arrays.

    quat_order describes every input root/object quaternion, including aliases.
    Group skill labels with convert_files before discarding them here.
    """
    if not isinstance(source, Mapping):
        raise ValueError("Each motion must be a dictionary")
    if quat_order not in ("xyzw", "wxyz"):
        raise ValueError("quat_order must be xyzw or wxyz")
    data = dict(source)
    root = _resolve(data, "root_pos", ("root_trans_offset",), _array)
    if root.ndim != 2 or root.shape[1] != 3 or root.shape[0] == 0:
        raise ValueError(f"root position must have shape (T,3), T >= 1; got {root.shape}")
    frames = len(root)
    if "fps" not in data:
        raise ValueError("Missing required field fps")
    fps_value = _array(data.pop("fps"), "fps")
    if fps_value.size != 1 or float(fps_value.reshape(-1)[0]) <= 0:
        raise ValueError("fps must be a positive scalar")
    fps = float(fps_value.reshape(-1)[0])
    quat = lambda value, key: _quaternion(value, key, frames, quat_order)[:, [3, 0, 1, 2]]
    root_rot = _resolve(data, "root_rot", (), quat)
    obj_rot = _resolve(data, "obj_rot", ("obj_rot_quat",), quat)
    obj_pos = _resolve(data, "obj_pos", (), lambda v, k: _sequence(v, k, frames, 3))
    if "dof" not in data:
        raise ValueError("Missing required field dof")
    dof = _array(data.pop("dof"), "dof")
    if dof.shape == (frames, 29, 1):
        dof = dof[..., 0]
    if dof.shape != (frames, 29):
        raise ValueError(f"dof has shape {dof.shape}; expected ({frames},29) or ({frames},29,1)")
    names = data.pop("dof_names", None)
    if names is not None:
        names = [str(name)[:-6] if str(name).endswith("_joint") else str(name) for name in names]
        if len(names) != 29 or set(names) != set(HUMANX_DOF_NAMES):
            raise ValueError("dof_names must contain each of the 29 G1 joint names exactly once")
        dof = dof[:, [names.index(name) for name in HUMANX_DOF_NAMES]]
    contacts = _bb_contacts(data, frames)
    hoop = _resolve(
        data, "hoop_pos_w", ("basket_pos",),
        lambda v, k: _sequence(v, k, frames, 3, constant=True),
        default=np.zeros((frames, 3), dtype=np.float32),
    )
    if "obj_freeze" in data or "freeze_obj" in data:
        freeze = _resolve(data, "freeze_obj", ("obj_freeze",), lambda v, k: _contact(v, k, frames))
        if np.any((freeze != 0) & (freeze != 1)):
            raise ValueError("obj_freeze/freeze_obj must contain only 0 or 1")
        data["freeze_obj"] = freeze.astype(np.int64)
    if "stand_frame" in data:
        data["stand_frame"] = _integer(data["stand_frame"], "stand_frame", minimum=0, maximum=frames - 1)
    for key in DERIVED_FIELDS | {"skill_label"}:
        data.pop(key, None)
    data.setdefault("push_available", np.zeros(frames, dtype=np.float32))
    data.setdefault("obj_push", np.zeros(frames, dtype=bool))
    for key in ("push_available", "obj_push", "release"):
        if key in data:
            data[key] = _frame_label(data[key], key, frames)
    data.update(
        fps=int(fps) if fps.is_integer() else fps,
        root_pos=root.copy(), root_rot=root_rot, obj_pos=obj_pos, obj_rot=obj_rot,
        dof=dof.copy(), dof_names=[name + "_joint" for name in HUMANX_DOF_NAMES],
        contact=contacts, contact_names=list(CONTACT_NAMES),
        hoop_pos_w=hoop[0].copy() if np.all(hoop == hoop[:1]) else hoop,
    )
    return data


def _skill_ids(source: Mapping, frames: int) -> np.ndarray:
    """Return original one-hot column IDs; -1 denotes unassigned frames."""
    if "skill_label" not in source:
        return np.full(frames, -1, dtype=np.int64)
    labels = _array(source["skill_label"], "skill_label")
    if labels.ndim != 2 or labels.shape[0] != frames or labels.shape[1] == 0:
        raise ValueError(f"skill_label must have shape ({frames}, K), K >= 1; got {labels.shape}")
    if np.any((labels != 0) & (labels != 1)) or np.any(labels.sum(axis=1) > 1):
        raise ValueError("skill_label must be one-hot or all-zero in every frame")
    return np.where(labels.any(axis=1), labels.argmax(axis=1), -1)


def _slice_motion(motion: dict, start: int, end: int, frames: int) -> dict:
    if start == 0 and end == frames:
        return motion
    result = {}
    for key, value in motion.items():
        if key in STATIC_FIELDS or (key == "hoop_pos_w" and np.shape(value) == (3,)):
            result[key] = value
        elif isinstance(value, np.ndarray) and value.ndim > 0 and len(value) == frames:
            result[key] = value[start:end].copy()
        elif isinstance(value, (list, tuple)) and len(value) == frames:
            result[key] = value[start:end]
        else:
            result[key] = value
    if "stand_frame" in result:
        result["stand_frame"] = int(np.clip(result["stand_frame"] - start, 0, end - start - 1))
    return result


def _add_motion(group: dict, name: str, motion: dict) -> None:
    key = name
    suffix = 2
    while key in group:
        key = f"{name}__{suffix}"
        suffix += 1
    group[key] = motion


def convert_files(
    input_paths: Sequence[Path], output: Path, *, quat_order: str = "xyzw",
) -> dict[Path, dict]:
    """Write flat single-motion dicts / batch lists, grouped by original skill ID.

    Return {path: {motion_name: motion}} for reporting; names are not an outer
    wrapper in the saved PKLs, since BB's editor and batch tools cannot read it.
    """
    paths = [Path(path).expanduser().resolve() for path in input_paths]
    output = Path(output).expanduser().resolve()
    if not paths:
        raise ValueError("At least one input file is required")
    if len(set(paths)) != len(paths):
        raise ValueError("Duplicate input paths: each file should be supplied only once")
    groups = {}
    for path in paths:
        for name, source in _motions(joblib.load(path), path):
            try:
                motion = convert_motion(source, quat_order=quat_order)
                frames = len(motion["root_pos"])
                ids = _skill_ids(source, frames)
                boundaries = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1, frames]
                base = name if len(paths) == 1 else f"{path.stem}__{name}"
                for start, end in zip(boundaries[:-1], boundaries[1:]):
                    skill = int(ids[start])
                    key = base if len(boundaries) == 2 else f"{base}__frames_{start}_{end}"
                    _add_motion(groups.setdefault(skill, {}), key, _slice_motion(motion, start, end, frames))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path} / motion {name}: {exc}") from exc

    outputs = {}
    directory = output.is_dir() or output.suffix.lower() != ".pkl"
    for skill, motions in sorted(groups.items()):
        label = "unlabelled" if skill == -1 else str(skill)
        if directory:
            target = output / f"skill_{label}.pkl"
        elif len(groups) == 1:
            target = output
        else:
            target = output.with_name(f"{output.stem}_skill_{label}{output.suffix}")
        target = target.resolve()
        if target in paths or (target.exists() and any(target.samefile(path) for path in paths)):
            raise ValueError("Output must not overwrite an input file")
        if target in outputs:
            raise ValueError(f"Different skills resolve to the same output file: {target}")
        outputs[target] = motions

    # Validate all inputs before writing; failed serialization leaves outputs intact.
    staged = []
    try:
        for target, motions in outputs.items():
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                dir=target.parent, prefix=f".{target.name}.", suffix=".tmp", delete=False,
            ) as handle:
                temporary = Path(handle.name)
            staged.append((temporary, target))
            payload = next(iter(motions.values())) if len(motions) == 1 else list(motions.values())
            joblib.dump(payload, temporary)
        for temporary, target in staged:
            os.replace(temporary, target)
    finally:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", type=Path, help="HumanX motion PKLs")
    parser.add_argument("-o", "--output", required=True, type=Path, help="BB .pkl path or output directory")
    parser.add_argument("--quat-order", choices=("xyzw", "wxyz"), default="xyzw",
                        help="Input root/object quaternion order (default: xyzw for HumanX)")
    args = parser.parse_args()
    try:
        outputs = convert_files(args.inputs, args.output, quat_order=args.quat_order)
    except (OSError, TypeError, ValueError, EOFError) as exc:
        parser.error(str(exc))
    for path, motions in outputs.items():
        frames = sum(len(motion["root_pos"]) for motion in motions.values())
        layout = "single motion dict" if len(motions) == 1 else "batch list"
        print(f"Wrote {len(motions)} motions, {frames} frames ({layout}) -> {path}")
    print(f"Exported {len(outputs)} skill file(s)")


if __name__ == "__main__":
    main()
