# Factory 每個任務的參數表

由 `task_table.py` 產生：每個值都來自 Isaac Lab 官方的 Factory 設定或函式（`source` 欄寫出處）。
- 產生方式：2026-10-04 在沒有 Isaac Lab 的筆電上，用最小的替身模組載入官方原始檔（IsaacLab `main` 的
  `direct/factory/`，和 v2.3.2 完全一樣）算出來的。
- 在訓練機器上要重算一次並比對：`./isaaclab.sh -p Isaac/hetero_factory/print_task_table.py --headless`。
- 單位：長度公尺、角度依官方欄位（`*_deg` 是度，`hand_init_orn` 是弧度）、四元數是 (w, x, y, z)。

| field | peg_insert | gear_mesh | nut_thread | source |
|---|---|---|---|---|
| num_keypoints | 4 | 4 | 4 | cfg: task.num_keypoints |
| episode_length_s | 10 | 20 | 30 | cfg: FactoryTask*Cfg.episode_length_s |
| hand_init_pos | (0, 0, 0.047) | (0, 0, 0.035) | (0, 0, 0.015) | cfg: task.hand_init_pos (relative to the fixed asset tip) |
| hand_init_pos_noise | (0.02, 0.02, 0.01) | (0.02, 0.02, 0.01) | (0.02, 0.02, 0.01) | cfg: task.hand_init_pos_noise |
| hand_init_orn | (3.1416, 0, 0) | (3.1416, 0, 0) | (3.1416, 0, 1.83) | cfg: task.hand_init_orn |
| hand_init_orn_noise | (0, 0, 0.785) | (0, 0, 0.785) | (0, 0, 0.26) | cfg: task.hand_init_orn_noise |
| unidirectional_rot | no | no | yes | cfg: task.unidirectional_rot |
| fixed_asset_init_pos_noise | (0.05, 0.05, 0.05) | (0.05, 0.05, 0.05) | (0.05, 0.05, 0.05) | cfg: task.fixed_asset_init_pos_noise |
| fixed_asset_init_orn_deg | 0 | 0 | 120 | cfg: task.fixed_asset_init_orn_deg |
| fixed_asset_init_orn_range_deg | 360 | 15 | 30 | cfg: task.fixed_asset_init_orn_range_deg |
| held_asset_pos_noise | (0.003, 0, 0.003) | (0.003, 0, 0.003) | (0, 0.003, 0.003) | cfg: task.held_asset_pos_noise |
| held_noise_z_one_sided | no | yes | no | replica: FactoryEnv.randomize_initial_state: gear z noise in [-1, 0] |
| keypoint_scale | 0.15 | 0.15 | 0.05 | cfg: task.keypoint_scale |
| keypoint_coef_baseline | (5, 4) | (5, 4) | (100, 2) | cfg: task.keypoint_coef_baseline |
| keypoint_coef_coarse | (50, 2) | (50, 2) | (500, 2) | cfg: task.keypoint_coef_coarse |
| keypoint_coef_fine | (100, 0) | (100, 0) | (1500, 0) | cfg: task.keypoint_coef_fine |
| action_penalty_ee_scale | 0 | 0 | 0 | cfg: task.action_penalty_ee_scale |
| action_grad_penalty_scale | 0 | 0 | 0 | cfg: task.action_grad_penalty_scale |
| success_height_threshold | 0.001 | 0.001 | 0.00075 | replica: FactoryEnv._get_curr_successes with task.success_threshold |
| engage_height_threshold | 0.0225 | 0.018 | 0.001 | replica: FactoryEnv._get_curr_successes with task.engage_threshold |
| success_check_rot | no | no | yes | replica: FactoryEnv._get_rewards: check_rot only for nut_thread |
| ee_success_yaw | 0 | 0 | 0 | cfg: task.ee_success_yaw |
| held_base_pos_local | (0, 0, 0) | (0.02025, 0, 0) | (0, 0, 0.01) | fn: factory_utils.get_held_base_pos_local |
| target_held_base_pos_local | (0, 0, 0) | (0.02025, 0, 0) | (0, 0, 0.032) | fn: factory_utils.get_target_held_base_pose (identity fixed pose) |
| fixed_tip_pos_local | (0, 0, 0.025) | (0.02025, 0, 0.025) | (0, 0, 0.035) | replica: FactoryEnv.randomize_initial_state: fixed_tip_pos_local |
| held_in_hand_pos | (0, 0, 0.032392) | (0.02025, 0, 0.0165) | (0, 0, 0.01) | fn: FactoryEnv.get_handheld_asset_relative_pose |
| held_in_hand_quat | (1, 0, 0, 0) | (1, 0, 0, 0) | (0.707107, 0, 0, -0.707107) | fn: FactoryEnv.get_handheld_asset_relative_pose |
| gripper_width | 0.00499125 | 0.01875 | 0.015 | replica: FactoryEnv._set_franka_to_default_pose: held diameter / 2 * 1.25 |
| has_flanking_gears | no | yes | no | replica: FactoryEnv: name == gear_mesh and task.add_flanking_gears |
| held_friction | 0.75 | 0.75 | 0.01 | cfg: task.held_asset_cfg.friction |
| fixed_friction | 0.75 | 0.75 | 0.75 | cfg: task.fixed_asset_cfg.friction |
| robot_friction | 0.75 | 0.75 | 0.75 | cfg: task.robot_cfg.friction |
| held_mass | 0.019 | 0.012 | 0.03 | cfg: task.held_asset_cfg.mass |
| fixed_mass | 0.05 | 0.05 | 0.05 | cfg: task.fixed_asset_cfg.mass |
