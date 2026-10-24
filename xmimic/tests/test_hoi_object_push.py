import os
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest


class HOIObjectPushTests(unittest.TestCase):
    def test_object_push_scheduling_and_state_submission(self):
        # Import Isaac Gym before torch in an isolated process, as required by Gym.
        # Exercise the real environment methods on CPU without constructing a simulator.
        root = Path(__file__).resolve().parents[1]
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([str(root), str(root / "isaac_utils"), env.get("PYTHONPATH", "")])
        env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
        env.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions")
        script = textwrap.dedent(
            """
            from humanoidverse.envs.humanx.hoi import LeggedRobotMotionTrackingHOI, LeggedRobotBase
            from types import SimpleNamespace as NS
            from unittest.mock import Mock, patch
            import torch

            def make_task(**overrides):
                cfg = dict(enabled=True, apply_in_eval=False, rate_hz=1.0,
                           cooldown_s=0.5, max_delta_v_xyz=[0.1, 0.2, 0.0])
                cfg.update(overrides)
                task = LeggedRobotMotionTrackingHOI.__new__(LeggedRobotMotionTrackingHOI)
                task.config = NS(domain_rand=NS(object_push=NS(**cfg)))
                task.num_envs, task.device, task.dt = 4, 'cpu', 0.02
                states = torch.arange(4 * 3 * 13, dtype=torch.float32).reshape(-1, 13)
                task.simulator = NS(all_root_states=states,
                                    object_root_state=states.view(4, 3, 13)[:, 1],
                                    robot_asset_idx=0, ball_asset_idx=1, support_asset_idx=2,
                                    dof_state=torch.zeros(4, 2))
                task._freeze_obj_buf = torch.zeros(4, dtype=torch.bool)
                task._visualize_motion_data_only = task.is_evaluating = False
                task.episode_length_buf = torch.ones(4, dtype=torch.long)
                task.need_to_refresh_envs = torch.zeros(4, dtype=torch.bool)
                task._init_object_push()
                if task._object_push_enabled:
                    task._object_push_allowed[:] = True
                return task

            empty_ids = torch.empty(0, dtype=torch.long)
            # Old configs and disabled configs are no-ops, including RNG state.
            for old_config in (False, True):
                task = make_task(enabled=False)
                if old_config:
                    task.config.domain_rand = NS()
                    task._init_object_push()
                with patch('torch.rand') as random:
                    task._apply_object_push(empty_ids)
                    random.assert_not_called()

            # Only an eligible ball changes; robot/support, pose and angular velocity stay intact.
            # First eligible control step is inside a 0.2-second window, with no initial cooldown.
            task = make_task()
            task._object_push_allowed[1] = False
            task._freeze_obj_buf[2] = True
            before = task.simulator.all_root_states.clone()
            with patch('torch.rand', side_effect=[torch.tensor([0.01]), torch.tensor([[1., 0., 0.5]])]):
                task._apply_object_push(torch.tensor([3]))
            expected = before.clone()
            expected[1, 7:10] += torch.tensor([0.1, -0.2, 0.0])
            torch.testing.assert_close(task.simulator.all_root_states, expected)
            assert task.need_to_refresh_envs.tolist() == [True, False, False, False]

            # Respect the full cooldown, including its exact end, without float countdown drift.
            task._object_push_allowed[:] = False
            task._object_push_allowed[0] = True
            task.episode_length_buf[:] = 25
            with patch('torch.rand') as random:
                task._apply_object_push(empty_ids)
                random.assert_not_called()
            task.episode_length_buf[:] = 26
            with patch('torch.rand', side_effect=[torch.tensor([0.0]), torch.ones(1, 3)]):
                task._apply_object_push(empty_ids)
            assert task._object_push_next_step[0] == 51

            # Reset clears cooldown and stale permission only for the selected env.
            task._state_delay_init_offset_s_robot = torch.ones(4)
            task._state_delay_init_offset_s_object = torch.ones(4)
            with patch.object(LeggedRobotBase, '_reset_buffers_callback'):
                task._reset_buffers_callback(torch.tensor([0]))
            assert task._object_push_next_step[0] == 0
            assert not task._object_push_allowed[0]

            # Closed windows, playback, evaluation, zero rate and just-reset envs never draw impulses.
            for condition in ('closed', 'playback', 'eval', 'zero_rate', 'reset'):
                task = make_task(rate_hz=0.0 if condition == 'zero_rate' else 1.0)
                if condition == 'closed': task._object_push_allowed[:] = False
                if condition == 'playback': task._visualize_motion_data_only = True
                if condition == 'eval': task.is_evaluating = True
                if condition == 'reset': task.episode_length_buf[:] = 0
                with patch('torch.rand') as random:
                    task._apply_object_push(empty_ids)
                    random.assert_not_called()

            # Explicit eval opt-in works. Bernoulli threshold depends on dt, not a per-step fixed rate.
            for dt, should_push in ((0.01, False), (0.02, True)):
                task = make_task(apply_in_eval=True)
                task.dt = dt
                task._init_object_push()
                task.is_evaluating = True
                task._object_push_allowed[0] = True
                with patch('torch.rand', side_effect=[torch.tensor([0.015]), torch.ones(1, 3)]):
                    task._apply_object_push(empty_ids)
                assert bool(task.need_to_refresh_envs[0]) == should_push

            for bad in (dict(rate_hz=-1), dict(rate_hz=float('nan')),
                        dict(cooldown_s=-1), dict(max_delta_v_xyz=[0.1, 0.2]),
                        dict(max_delta_v_xyz=[0.1, -0.1, 0]),
                        dict(max_delta_v_xyz=[0.1, float('inf'), 0])):
                try:
                    make_task(**bad)
                except ValueError:
                    pass
                else:
                    raise AssertionError('accepted invalid object_push config: ' + str(bad))

            # Exercise the actual post-step call site and actor submission with simulator I/O stubbed.
            task = make_task()
            task.config.robot = NS(proj=False)
            task.config.normalization = NS(clip_observations=100)
            task.reset_buf = torch.tensor([0, 1, 0, 0])
            task._object_push_allowed[:] = torch.tensor([True, True, False, False])
            task._refresh_sim_tensors = Mock()
            task._update_counters_each_step = Mock()
            task._pre_compute_observations_callback = Mock()
            task._update_tasks_callback = Mock()
            task._check_termination = Mock()
            task._compute_reward = Mock()
            task._post_compute_observations_callback = Mock()
            task.history_handler = NS(history={})
            task.obs_buf_dict, task.extras, task.log_dict = {}, {}, {}
            task.viewer = None
            task.reset_envs_idx = lambda ids: task.episode_length_buf.index_fill_(0, ids, 0)
            submitted = []
            task.simulator.set_actor_root_state_tensor = lambda ids, states: submitted.append((ids.clone(), states.clone()))
            task.simulator.set_dof_state_tensor = Mock()
            task._compute_observations = lambda: task.obs_buf_dict.update(ball_vel=task.simulator.object_root_state[:, 7:10].clone())
            before = task.simulator.all_root_states.clone()
            with patch('torch.rand', side_effect=[torch.tensor([0.0]), torch.ones(1, 3)]):
                task._hoi_post_physics_step()
            assert len(submitted) == 1
            assert set(submitted[0][0].tolist()) == {0, 1, 2}
            expected = before.clone()
            expected[1, 7:10] += torch.tensor([0.1, 0.2, 0.0])
            torch.testing.assert_close(submitted[0][1], expected)
            torch.testing.assert_close(task.obs_buf_dict['ball_vel'][0], expected[1, 7:10])
            assert task._pre_compute_observations_callback.call_count == 2
            """
        )
        result = subprocess.run([sys.executable, "-c", script], cwd=str(root), env=env,
                                capture_output=True, text=True, timeout=120)
        if result.returncode:
            self.fail(result.stderr + result.stdout)


if __name__ == "__main__":
    unittest.main()
