import os
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest


class HOIRotationRewardTests(unittest.TestCase):
    def test_rotation_rewards_use_shortest_angle_and_preserve_options(self):
        # Isolate Isaac Gym's requirement to be imported before torch. No
        # simulator is constructed: call the real reward methods on CPU tensors.
        root = Path(__file__).resolve().parents[1]
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            [str(root), str(root / "isaac_utils"), env.get("PYTHONPATH", "")]
        )
        env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
        env.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions")
        script = textwrap.dedent(
            """
            from humanoidverse.envs.humanx.hoi import LeggedRobotMotionTrackingHOI
            import math
            from types import SimpleNamespace
            import torch

            task = LeggedRobotMotionTrackingHOI.__new__(LeggedRobotMotionTrackingHOI)
            sigma = SimpleNamespace(teleop_body_rot=1.0, ball_rotation=1.0)
            rewards = SimpleNamespace(
                reward_tracking_sigma=sigma,
                anchor_centered_body_position=False,
                ball_rotation_consider_yaw=True,
            )
            task.config = SimpleNamespace(rewards=rewards)
            task._get_anchor_body_index = lambda: 0

            # Known z-axis rotations, including the shortest path past 180 deg.
            angles = torch.tensor(
                [0.0, 1e-4, math.pi / 2, math.pi, 3 * math.pi / 2],
                dtype=torch.float64,
            )
            q = torch.zeros((len(angles), 4), dtype=torch.float64)
            q[:, 2] = torch.sin(angles / 2)
            q[:, 3] = torch.cos(angles / 2)
            expected = torch.tensor(
                [1.0, math.exp(-1e-8), math.exp(-math.pi**2 / 4),
                 math.exp(-math.pi**2), math.exp(-math.pi**2 / 4)],
                dtype=torch.float64,
            )
            for scale in (1.0, -1.0, 2.5, -0.4):
                task.dif_global_body_rot = (q * scale).unsqueeze(1).repeat(1, 2, 1)
                task.dif_global_ball_rot = q * scale
                original_body = task.dif_global_body_rot.clone()
                original_ball = task.dif_global_ball_rot.clone()
                for method in (
                    task._reward_teleop_body_rotation_extend,
                    task._reward_teleop_anchor_rotation,
                    task._reward_ball_rotation,
                ):
                    torch.testing.assert_close(method(), expected, rtol=1e-6, atol=1e-12)
                assert torch.equal(task.dif_global_body_rot, original_body)
                assert torch.equal(task.dif_global_ball_rot, original_ball)

            # Average squared angles across bodies, not quaternion components
            # or angles before squaring: identity plus 90 deg.
            task.dif_global_body_rot = q[[0, 2]].unsqueeze(0)
            torch.testing.assert_close(
                task._reward_teleop_body_rotation_extend(),
                torch.tensor([math.exp(-math.pi**2 / 8)], dtype=q.dtype),
            )

            # Anchor-centered mode uses its own errors and excludes the anchor.
            rewards.anchor_centered_body_position = True
            task._get_anchor_centered_dif_global_body_rot = lambda: q[[3, 2]].unsqueeze(0)
            torch.testing.assert_close(
                task._reward_teleop_body_rotation_extend(), expected[2:3]
            )
            sigma.teleop_anchor_rot = 2.0
            task.dif_global_body_rot = q[2:3].unsqueeze(1)
            torch.testing.assert_close(
                task._reward_teleop_anchor_rotation(),
                torch.tensor([math.exp(-math.pi**2 / 8)], dtype=q.dtype),
            )

            # Ignore yaw only when requested; roll must still be penalized.
            rewards.ball_rotation_consider_yaw = False
            task.dif_global_ball_rot = q[2:3].clone()
            torch.testing.assert_close(task._reward_ball_rotation(), torch.ones(1, dtype=q.dtype))
            task.dif_global_ball_rot = q[2:3][:, [2, 1, 0, 3]].clone()
            torch.testing.assert_close(task._reward_ball_rotation(), expected[2:3])
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=str(root),
            text=True,
            capture_output=True,
            env=env,
            timeout=120,
        )
        if result.returncode != 0:
            self.fail(result.stderr or result.stdout)


if __name__ == "__main__":
    unittest.main()
