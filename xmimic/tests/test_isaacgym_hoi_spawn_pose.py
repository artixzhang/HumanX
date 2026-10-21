import os
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest


class IsaacGymHOISpawnPoseTests(unittest.TestCase):
    def test_robot_spawn_preserves_base_position_above_each_env_origin(self):
        root = Path(__file__).resolve().parents[1]
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            [str(root), str(root / "isaac_utils"), env.get("PYTHONPATH", "")]
        )
        env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
        env.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions")
        script = textwrap.dedent(
            """
            from types import SimpleNamespace
            from unittest.mock import Mock, call, patch
            from humanoidverse.simulator.isaacgym import isaacgym_hoi
            import torch

            sim = isaacgym_hoi.IsaacGym.__new__(isaacgym_hoi.IsaacGym)
            sim.device = "cpu"
            sim.env_origins = torch.tensor([
                [0.0, 0.0, 0.0],
                [12.0, -7.0, 0.35],
            ])
            sim.base_init_state = torch.tensor([
                0.15, -0.2, 0.8,
                0.0, 0.0, 0.0, 1.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ])
            original_origins = sim.env_origins.clone()
            original_base = sim.base_init_state.clone()
            xy_noise = torch.tensor([[0.25], [-0.4]])
            sim.env_config = SimpleNamespace(robot=SimpleNamespace(
                asset=SimpleNamespace(robot_type="g1_29dof", self_collisions=0),
            ))
            sim.robot_config = SimpleNamespace(proj=False)
            sim.robot_asset = object()
            sim.envs = []
            sim.robot_handles = []
            captured_positions = []

            def create_actor(env_ptr, asset, pose, name, env_id, collision_filter,
                             segmentation_id):
                captured_positions.append(torch.tensor([
                    pose.p.x, pose.p.y, pose.p.z,
                ]))
                return "robot_" + str(env_id)

            # Exercise the real method with only the simulator boundary stubbed.
            sim.gym = Mock()
            sim.gym.create_actor.side_effect = create_actor
            sim.gym.get_actor_rigid_body_names.return_value = ["pelvis"]
            sim._process_rigid_shape_props = lambda props, env_id: props
            sim._process_dof_props = lambda props, env_id: props
            sim._process_rigid_body_props = lambda props, env_id: props
            sim._build_objects = Mock()

            with patch.object(isaacgym_hoi, "torch_rand_float",
                              return_value=xy_noise) as sample_noise:
                for env_id in range(len(sim.env_origins)):
                    sim._build_each_env(env_id, "env_" + str(env_id))
                    expected = (
                        original_origins[env_id]
                        + original_base[:3]
                        + torch.tensor([0.25, -0.4, 0.0])
                    )
                    torch.testing.assert_close(captured_positions[env_id], expected)
                    # Adding spawn jitter must not move the terrain origin or
                    # change the configured base pose for subsequent environments.
                    torch.testing.assert_close(sim.env_origins, original_origins)
                    torch.testing.assert_close(sim.base_init_state, original_base)

                assert sample_noise.call_args_list == [
                    call(-1.0, 1.0, (2, 1), device="cpu"),
                    call(-1.0, 1.0, (2, 1), device="cpu"),
                ]

            assert sim.envs == ["env_0", "env_1"]
            assert sim.robot_handles == ["robot_0", "robot_1"]
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            text=True,
            capture_output=True,
            env=env,
            cwd=str(root),
            timeout=120,
        )
        if result.returncode != 0:
            self.fail(result.stderr or result.stdout)


if __name__ == "__main__":
    unittest.main()
