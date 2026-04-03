
from legged_gym.envs.base.legged_robot import LeggedRobot

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import torch
from legged_gym.envs.g1.g1_style import G1StyleModule

class G1Robot(LeggedRobot):
    def _init_foot(self):
        self.feet_num = len(self.feet_indices)
        
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        self.rigid_body_states_view = self.rigid_body_states.view(self.num_envs, -1, 13)
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]
        
    def _init_buffers(self):
        super()._init_buffers()
        self._init_foot()
        enabled = self.cfg.rewards.enable_style_discriminator
        self.style_module = G1StyleModule(self) if enabled else None
        if not enabled:
            scales = self.cfg.rewards.scales
            scales.gait_naturalness = 0.0
            scales.body_posture = 0.0
            scales.arm_coordination = 0.0
            scales.arm_swing_phase = 0.0
            scales.step_characteristics = 0.0
            scales.movement_fluency = 0.0

    def _create_envs(self):
        super()._create_envs()

        self.hip_dof_indices = []
        for i in range(len(self.dof_names)):
            if any([s in self.dof_names[i] for s in self.cfg.asset.hip_dof_name]):
                self.hip_dof_indices.append(i)
        self.hip_dof_indices = torch.tensor(self.hip_dof_indices, dtype=torch.long, device=self.device, requires_grad=False)
        
        self.shoulder_dof_indices = []
        for i in range(len(self.dof_names)):
            if any([s in self.dof_names[i] for s in self.cfg.asset.shoulder_dof_name]):
                self.shoulder_dof_indices.append(i)
        self.shoulder_dof_indices = torch.tensor(self.shoulder_dof_indices, dtype=torch.long, device=self.device, requires_grad=False)

        self.hip_pitch_dof_indices = []
        for i in range(len(self.dof_names)):
            if any([s in self.dof_names[i] for s in self.cfg.asset.hip_pitch_dof_name]):
                self.hip_pitch_dof_indices.append(i)
        self.hip_pitch_dof_indices = torch.tensor(self.hip_pitch_dof_indices, dtype=torch.long, device=self.device, requires_grad=False)

        self._hip_pitch_id_list = self.hip_pitch_dof_indices.detach().cpu().tolist() if self.hip_pitch_dof_indices.numel() > 0 else []
        self._shoulder_id_list = self.shoulder_dof_indices.detach().cpu().tolist() if self.shoulder_dof_indices.numel() > 0 else []
        self._shoulder_sync_enabled = len(self._hip_pitch_id_list) == 2 and len(self._shoulder_id_list) == 2


    def update_feet_state(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]
        
    def _post_physics_step_callback(self):
        self.update_feet_state()

        period = 0.8
        offset = 0.5
        self.phase = (self.episode_length_buf * self.dt) % period / period
        self.phase_left = self.phase
        self.phase_right = (self.phase + offset) % 1
        self.leg_phase = torch.cat([self.phase_left.unsqueeze(1), self.phase_right.unsqueeze(1)], dim=-1)
        
        if self.style_module:
            self.style_module.update()
        
        return super()._post_physics_step_callback()

    def _compute_torques(self, actions):
        if getattr(self, "_shoulder_sync_enabled", False):
            left_hip_idx, right_hip_idx = self._hip_pitch_id_list
            left_shoulder_idx, right_shoulder_idx = self._shoulder_id_list
            gain = self.cfg.control.shoulder_coupling_gain
            if gain != 0.0:
                actions[:, right_shoulder_idx] = actions[:, left_hip_idx] * gain + 0.8
                actions[:, left_shoulder_idx] = actions[:, right_hip_idx] * gain + 0.8
            else:
                actions[:, right_shoulder_idx] = 0.0
                actions[:, left_shoulder_idx] = 0.0

        return super()._compute_torques(actions)
    
    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if len(env_ids) == 0:
            return
        if self.style_module:
            self.style_module.reset(env_ids)

    def _style_reward(self, accessor):
        if not self.style_module:
            return torch.zeros(self.num_envs, device=self.device)
        method = getattr(self.style_module, accessor, None)
        if method is None:
            raise AttributeError(f"Style module missing accessor: {accessor}")
        return method()
    
    def _reward_gait_naturalness(self):
        """Gait naturalness: derived from HumanLikeGaitReward periodicity and speed-consistency features."""
        return self._style_reward('reward_gait_naturalness')
    
    def _reward_body_posture(self):
        """Posture stability: focuses on body sway balance while ignoring height fluctuations."""
        return self._style_reward('reward_body_posture')

    def _reward_arm_coordination(self):
        """Arm coordination: derived from arm-swing balance and arm–leg phase-difference features."""
        return self._style_reward('reward_arm_coordination')

    def _reward_arm_swing_phase(self):
        """Arm-swing reward based on HumanLikeGaitReward phase-difference features."""
        return self._style_reward('reward_arm_swing_phase')
    
    def _reward_step_characteristics(self):
        """Step length and cadence consistency reward (from human-like feature extraction)."""
        return self._style_reward('reward_step_characteristics')

    def _reward_movement_fluency(self):
        """Movement fluency: combines speed smoothness and energy-utilization features."""
        return self._style_reward('reward_movement_fluency')
    
    def _reward_contact(self):
        res = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        for i in range(self.feet_num):
            is_stance = self.leg_phase[:, i] < 0.55
            contact = self.contact_forces[:, self.feet_indices[i], 2] > 1
            res += ~(contact ^ is_stance)
        return res

    def _reward_alive(self):
        # Reward for staying alive
        return 1.0
    
    def _reward_contact_no_vel(self):
        # Penalize contact with no velocity
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        contact_feet_vel = self.feet_vel * contact.unsqueeze(-1)
        penalize = torch.square(contact_feet_vel[:, :, :3])
        return torch.sum(penalize, dim=(1,2))
    
    def _reward_hip_pos(self):
        return torch.sum(torch.square(self.dof_pos[:, self.hip_dof_indices] - self.default_dof_pos[:, self.hip_dof_indices]), dim=1)

    def _reward_feet_edge(self):
        feet_pos_xy = ((self.rigid_body_states_view[:, self.feet_indices, :2] + self.terrain.cfg.border_size) / self.cfg.terrain.horizontal_scale).round().long()  # (num_envs, num_feet, 2)
        feet_pos_xy[..., 0] = torch.clip(feet_pos_xy[..., 0], 0, self.x_edge_mask.shape[0]-1)
        feet_pos_xy[..., 1] = torch.clip(feet_pos_xy[..., 1], 0, self.x_edge_mask.shape[1]-1)
        feet_at_edge = self.x_edge_mask[feet_pos_xy[..., 0], feet_pos_xy[..., 1]]
        self.feet_at_edge = self.contact_filt & feet_at_edge
        rew = (self.terrain_levels > 3) * torch.sum(self.feet_at_edge, dim=-1)
        return rew
    
    def _reward_stride_symmetry(self):
        base_x = self.base_pos[:, 0].unsqueeze(1)
        rel_stride = self.feet_pos[:, :, 0] - base_x
        stride_balance = torch.square(torch.sum(rel_stride, dim=1))

        if getattr(self, "_hip_pitch_id_list", None) and len(self._hip_pitch_id_list) == 2:
            hip_pitch = self.dof_pos[:, self.hip_pitch_dof_indices]
            stride_balance += 0.2 * torch.square(torch.sum(hip_pitch, dim=1))

        return stride_balance
    
    def _reward_roll_stability(self):
        roll = self.rpy[:, 0]
        lateral_vel = self.base_lin_vel[:, 1]
        sway = torch.square(roll) + 0.1 * torch.square(lateral_vel)
        return sway

    def _reward_pitch_upright(self):
        forward_pitch = torch.clamp(-self.pitch, min=0.0)
        speed_factor = 1.0 + 0.5 * torch.relu(self.base_lin_vel[:, 0])
        return torch.square(forward_pitch) * speed_factor