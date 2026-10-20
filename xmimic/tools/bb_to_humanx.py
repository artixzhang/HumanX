#!/usr/bin/env python3
"""Convert BB G1 basketball PKLs into a HumanX HOI motion dictionary.

Usage (from the repository root)::

    python xmimic/tools/bb_to_humanx.py shot.pkl -o shot_humanx.pkl
    python xmimic/tools/bb_to_humanx.py shot.pkl dribble.pkl -o merged.pkl
    python xmimic/tools/bb_to_humanx.py shot.pkl dribble.pkl -o merged.pkl --num-skills 2

Each input FILE is one skill, even if it contains a batch of many trajectories.
Skills occupy columns 0, 1, ... in input order; unused columns remain zero.
The default width, 42, matches hoi_teacher_unified_tracking.yaml. A single input
uses only column 0, still with width 42 (use --num-skills 1 for width 1). Set the
training observation's skill_label dimension to the same width when overriding.

Inputs may be one motion dict, a list/tuple of motion dicts, or a named motion
dict. Outputs always have {motion_name: motion_dict} structure. Requires numpy,
scipy and joblib; no simulator is needed.

Conversions:
  root_pos -> root_trans_offset; root_rot and obj_rot: WXYZ -> XYZW;
  obj_rot -> obj_rot_quat; dof -> (T,29,1), reordered by dof_names if available;
  pose_aa is rebuilt as (T,30,3) from root rotation and G1 joint axes;
  four-column contact -> lfoot_contact, rfoot_contact, max(left/right hand);
  hoop_pos_w -> basket_pos (broadcast to T frames).

Consumed aliases/contact_names and obsolete obj_rot_euler are removed. Other
fields (including hoop_pos_cyl, push_available, obj_push and release) survive.
release is independent of contact and object freezing, so is never used to
derive either. Missing foot contacts use HumanX's unknown value 0.5; missing
object velocities use finite differences; obj_freeze defaults to zero;
stand_frame defaults to the last frame. Positions, FPS and frame counts are
preserved: no resampling, coordinate translation or height correction occurs.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

import joblib
import numpy as np
from scipy.spatial.transform import Rotation

if __package__:
    from .convert_xgen_hoi_npz import G1_29DOF_AXES, HUMANX_DOF_NAMES
else:
    from convert_xgen_hoi_npz import G1_29DOF_AXES, HUMANX_DOF_NAMES


DEFAULT_NUM_SKILLS = 42
CONTACT_NAMES = (
    "left_foot_ground", "right_foot_ground", "left_hand_ball", "right_hand_ball"
)
_MISSING = object()


def _array(value, key: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{key} contains non-finite values")
    return result


def _sequence(value, key: str, frames: int, width: int, *, constant=False) -> np.ndarray:
    result = _array(value, key)
    if width == 1 and result.shape == (frames,):
        result = result[:, None]
    if constant:
        if width == 1 and result.ndim == 0:
            result = result.reshape(1, 1)
        if result.shape == (width,):
            result = result[None, :]
        if result.shape == (1, width):
            result = np.repeat(result, frames, axis=0)
    if result.shape != (frames, width):
        raise ValueError(f"{key} has shape {result.shape}; expected ({frames}, {width})")
    return result.copy()


def _resolve(data: dict, key: str, aliases: tuple, parse, default=_MISSING):
    """Consume aliases and reject conflicting duplicate representations."""
    values = [(name, parse(data.pop(name), name)) for name in (key, *aliases) if name in data]
    if not values:
        if default is _MISSING:
            raise ValueError(f"Missing required field {key} (aliases: {', '.join(aliases) or 'none'})")
        return default
    name, result = values[0]
    for other_name, other in values[1:]:
        if result.shape != other.shape or not np.allclose(result, other, rtol=1e-5, atol=1e-6):
            raise ValueError(f"Conflicting fields {name} and {other_name}")
    return result


def _integer(value, key: str, *, minimum: int, maximum: int | None = None) -> int:
    number = np.asarray(value)
    if number.size != 1:
        raise ValueError(f"{key} must be an integer scalar")
    scalar = float(number.reshape(-1)[0])
    if not np.isfinite(scalar) or not scalar.is_integer() or scalar < minimum:
        raise ValueError(f"{key} must be an integer >= {minimum}, got {value!r}")
    result = int(scalar)
    if maximum is not None and result > maximum:
        raise ValueError(f"{key} must be <= {maximum}, got {result}")
    return result


def _quaternion(value, key: str, frames: int, order: str) -> np.ndarray:
    result = _sequence(value, key, frames, 4)
    if order == "wxyz":
        result = result[:, [1, 2, 3, 0]]
    norms = np.linalg.norm(result.astype(np.float64), axis=1, keepdims=True)
    if np.any(norms < 1e-12):
        raise ValueError(f"{key} contains a zero-norm quaternion")
    return (result / norms).astype(np.float32)


def _contact(value, key: str, frames: int) -> np.ndarray:
    result = _sequence(value, key, frames, 1, constant=True)
    if np.any((result < 0) | (result > 1)):
        raise ValueError(f"{key} must contain values in [0, 1]")
    return result


def _contacts(data: dict, frames: int) -> dict:
    if "contact" not in data:
        raise ValueError("Missing required field contact; cannot infer physical hand contact from release")
    raw = _array(data.pop("contact"), "contact")
    names = data.pop("contact_names", None)
    feet = {}
    if raw.shape == (frames, 4):
        names = list(CONTACT_NAMES if names is None else names)
        if len(names) != 4 or set(names) != set(CONTACT_NAMES):
            raise ValueError(f"contact_names must contain exactly {CONTACT_NAMES}")
        raw = raw[:, [names.index(name) for name in CONTACT_NAMES]]
        if np.any((raw < 0) | (raw > 1)):
            raise ValueError("contact must contain values in [0, 1]")
        feet = {"lfoot_contact": raw[:, :1], "rfoot_contact": raw[:, 1:2]}
        hand_contact = raw[:, 2:4].max(axis=1, keepdims=True)
    else:
        hand_contact = _contact(raw, "contact", frames)
    result = {"contact": hand_contact.copy()}
    for key in ("lfoot_contact", "rfoot_contact"):
        value = feet.get(key, np.full((frames, 1), 0.5, dtype=np.float32))
        if key in data:
            explicit = _contact(data.pop(key), key, frames)
            if key in feet and not np.allclose(value, explicit):
                raise ValueError(f"{key} conflicts with four-column contact")
            value = explicit
        result[key] = value.copy()
    return result


def convert_motion(
    source: dict, *, skill_index: int = 0, num_skills: int = DEFAULT_NUM_SKILLS,
    quat_order: str = "wxyz",
) -> dict:
    """Convert one G1 motion without modifying the source dictionary or arrays.

    quat_order describes ALL input root/object quaternions, including an input
    already named obj_rot_quat. Use 'xyzw' when reprocessing HumanX data.
    """
    if not isinstance(source, Mapping):
        raise ValueError("Each motion must be a dictionary")
    num_skills = _integer(num_skills, "num_skills", minimum=1)
    skill_index = _integer(skill_index, "skill_index", minimum=0, maximum=num_skills - 1)
    if quat_order not in ("wxyz", "xyzw"):
        raise ValueError("quat_order must be wxyz or xyzw")
    data = dict(source)
    root = _resolve(data, "root_trans_offset", ("root_pos",), _array)
    if root.ndim != 2 or root.shape[1] != 3 or root.shape[0] < 3:
        raise ValueError(f"root position must have shape (T,3), T >= 3; got {root.shape}")
    frames = root.shape[0]
    if "fps" not in data:
        raise ValueError("Missing required field fps")
    fps_value = _array(data.pop("fps"), "fps")
    if fps_value.size != 1 or float(fps_value.reshape(-1)[0]) <= 0:
        raise ValueError("fps must be a positive scalar")
    fps = float(fps_value.reshape(-1)[0])
    quat = lambda value, key: _quaternion(value, key, frames, quat_order)
    root_rot = _resolve(data, "root_rot", (), quat)
    obj_rot = _resolve(data, "obj_rot_quat", ("obj_rot",), quat)
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
    pose_aa = np.zeros((frames, 30, 3), dtype=np.float32)
    pose_aa[:, 0] = Rotation.from_quat(root_rot).as_rotvec()
    pose_aa[:, 1:] = dof[:, :, None] * G1_29DOF_AXES[None]

    contacts = _contacts(data, frames)
    basket_pos = _resolve(
        data, "basket_pos", ("hoop_pos_w",),
        lambda v, k: _sequence(v, k, frames, 3, constant=True),
        default=np.zeros((frames, 3), dtype=np.float32),
    )
    freeze = _resolve(
        data, "obj_freeze", ("freeze_obj",), lambda v, k: _contact(v, k, frames),
        default=np.zeros((frames, 1), dtype=np.float32),
    )
    if np.any((freeze != 0) & (freeze != 1)):
        raise ValueError("obj_freeze must contain only 0 or 1")

    # Match HumanX's world-space backward differences, repeating the first interval.
    obj_vel = np.diff(obj_pos, axis=0) * fps
    obj_vel = np.concatenate((obj_vel[:1], obj_vel), axis=0)
    rotations = Rotation.from_quat(obj_rot)
    obj_ang_vel = (rotations[1:] * rotations[:-1].inv()).as_rotvec() * fps
    obj_ang_vel = np.concatenate((obj_ang_vel[:1], obj_ang_vel), axis=0).astype(np.float32)
    for key, fallback in (("obj_vel", obj_vel), ("obj_ang_vel", obj_ang_vel)):
        data[key] = _sequence(data[key], key, frames, 3) if key in data else fallback
    stand_frame = _integer(data.pop("stand_frame", frames - 1), "stand_frame", minimum=0, maximum=frames - 1)
    skill_label = np.zeros((frames, num_skills), dtype=np.float32)
    skill_label[:, skill_index] = 1
    # HumanX uses the quaternion; the Euler representation is obsolete.
    data.pop("obj_rot_euler", None)
    data.update(
        fps=int(fps) if fps.is_integer() else fps,
        root_trans_offset=root.copy(), root_rot=root_rot, dof=dof[:, :, None].copy(),
        dof_names=[name + "_joint" for name in HUMANX_DOF_NAMES], pose_aa=pose_aa,
        obj_pos=obj_pos, obj_rot_quat=obj_rot, **contacts, basket_pos=basket_pos,
        obj_freeze=freeze.astype(np.int64), stand_frame=stand_frame, skill_label=skill_label,
    )
    return data


def _motions(payload, path: Path):
    if isinstance(payload, Mapping):
        if "root_pos" in payload or "root_trans_offset" in payload:
            yield "0", payload
        elif payload:
            yield from ((str(name), motion) for name, motion in payload.items())
        else:
            raise ValueError(f"{path}: input contains no motions")
    elif isinstance(payload, (list, tuple)) and payload:
        yield from ((f"{index:04d}", motion) for index, motion in enumerate(payload))
    else:
        raise ValueError(f"{path}: expected a motion dict, nonempty list/tuple, or named motion dict")


def convert_files(
    input_paths: Sequence[Path], output: Path, *, num_skills: int = DEFAULT_NUM_SKILLS,
    quat_order: str = "wxyz",
) -> dict:
    """Merge input files, assigning one skill per file in the supplied order."""
    paths = [Path(path).expanduser().resolve() for path in input_paths]
    output = Path(output).expanduser().resolve()
    num_skills = _integer(num_skills, "num_skills", minimum=1)
    if not paths:
        raise ValueError("At least one input file is required")
    if len(set(paths)) != len(paths):
        raise ValueError("Duplicate input paths: each distinct file should be supplied only once")
    if len(paths) > num_skills:
        raise ValueError(f"{len(paths)} input files require at least {len(paths)} skills; num_skills={num_skills}")
    if output in paths or (output.exists() and any(output.samefile(path) for path in paths if path.exists())):
        raise ValueError("Output must not overwrite an input file")
    converted = {}
    for skill_index, path in enumerate(paths):
        count = 0
        for name, motion in _motions(joblib.load(path), path):
            key_base = f"{path.stem}__{name}"
            key = key_base
            suffix = 2
            while key in converted:
                key = f"{key_base}__{suffix}"
                suffix += 1
            try:
                converted[key] = convert_motion(motion, skill_index=skill_index, num_skills=num_skills, quat_order=quat_order)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path} / motion {name}: {exc}") from exc
            count += 1
        print(f"skill {skill_index} / {num_skills}: {path} -> {count} motions")
    output.parent.mkdir(parents=True, exist_ok=True)
    # A failed write leaves any existing output intact.
    with tempfile.NamedTemporaryFile(dir=output.parent, prefix=f".{output.name}.", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        joblib.dump(converted, temporary)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return converted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", type=Path, help="BB PKLs, one skill per file in input order")
    parser.add_argument("-o", "--output", required=True, type=Path, help="Merged HumanX motion PKL")
    parser.add_argument("--num-skills", "--skill-dim", "--label-dim", type=int, default=DEFAULT_NUM_SKILLS,
                        help="One-hot width (default: 42); must match the training observation dimension")
    parser.add_argument("--quat-order", choices=("wxyz", "xyzw"), default="wxyz",
                        help="Input root/object quaternion order (default: wxyz for BB)")
    args = parser.parse_args()
    try:
        motions = convert_files(args.inputs, args.output, num_skills=args.num_skills, quat_order=args.quat_order)
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    frames = sum(len(motion["root_trans_offset"]) for motion in motions.values())
    print(f"Wrote {len(motions)} motions, {frames} frames, {len(args.inputs)} active skills "
          f"(width {args.num_skills}) -> {args.output}")


if __name__ == "__main__":
    main()
