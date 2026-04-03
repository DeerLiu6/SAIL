import torch

from legged_gym.utils.style_discriminator import HumanLikeGaitReward


class G1StyleModule:

	def __init__(self, robot):
		self.robot = robot
		self.device = robot.device
		self.num_envs = robot.num_envs
		self.human_like_gait_reward = HumanLikeGaitReward(device=self.device)
		self._init_buffers()

	def _init_buffers(self):
		self.gait_history_length = 10
		self.gait_history = torch.zeros(self.num_envs, self.gait_history_length, 6, device=self.device)

		self.pose_history_length = 5
		self.pose_history = torch.zeros(self.num_envs, self.pose_history_length, 4, device=self.device)

		self.arm_history_length = 8
		self.arm_history = torch.zeros(self.num_envs, self.arm_history_length, 4, device=self.device)

		self.step_length_history = torch.zeros(self.num_envs, 3, device=self.device)
		self.step_frequency_history = torch.zeros(self.num_envs, 3, device=self.device)
		self.latest_step_length_lr = torch.zeros(self.num_envs, 2, device=self.device)
		self.latest_step_frequency_lr = torch.zeros(self.num_envs, 2, device=self.device)
		self.last_step_time_lr = torch.zeros(self.num_envs, 2, device=self.device)
		self.last_step_pos_lr = torch.zeros(self.num_envs, 2, 2, device=self.device)

		self._style_reward_cache = torch.zeros(self.num_envs, device=self.device)
		self._style_component_cache = {}
		self._style_detail_cache = {}
		self._style_cache_valid = False
		self._cache_counter = 0

	def update(self):
		with torch.no_grad():
			robot = self.robot
			base_height = robot.root_states[:, 2:3] - torch.mean(robot.feet_pos[:, :, 2], dim=1, keepdim=True)

			current_gait_data = torch.cat([
				robot.root_states[:, :2],
				robot.root_states[:, 7:9],
				robot.leg_phase[:, 0:1],
				base_height,
			], dim=1)
			self.gait_history = torch.roll(self.gait_history, -1, dims=1)
			self.gait_history[:, -1] = current_gait_data

			current_pose_data = torch.cat([robot.rpy, base_height], dim=1)
			self.pose_history = torch.roll(self.pose_history, -1, dims=1)
			self.pose_history[:, -1] = current_pose_data

			current_arm_data = torch.cat([
				robot.dof_pos[:, robot.shoulder_dof_indices],
				robot.leg_phase,
			], dim=1)
			self.arm_history = torch.roll(self.arm_history, -1, dims=1)
			self.arm_history[:, -1] = current_arm_data

			self._cache_counter += 1
			if self._cache_counter % 3 == 0:
				self._detect_and_update_step_features()

		self._style_cache_valid = False
		self._evaluate_human_like_reward()

	def _detect_and_update_step_features(self):
		robot = self.robot
		current_time = robot.episode_length_buf * robot.dt
		current_pos = robot.root_states[:, :2]

		left_stance_transition = (robot.leg_phase[:, 0] >= 0.5) & (robot.leg_phase[:, 0] < 0.55)
		right_stance_transition = (robot.leg_phase[:, 1] >= 0.5) & (robot.leg_phase[:, 1] < 0.55)

		if torch.any(left_stance_transition):
			step_lengths_L = torch.norm(current_pos - self.last_step_pos_lr[:, 0, :], dim=1)
			step_durations_L = current_time - self.last_step_time_lr[:, 0]
			step_freqs_L = torch.where(
				(step_durations_L > 0) & (self.last_step_time_lr[:, 0] > 0),
				1.0 / step_durations_L,
				torch.zeros_like(step_durations_L),
			)

			self.latest_step_length_lr[left_stance_transition, 0] = step_lengths_L[left_stance_transition]
			self.latest_step_frequency_lr[left_stance_transition, 0] = step_freqs_L[left_stance_transition]

			self.step_length_history[left_stance_transition] = torch.roll(
				self.step_length_history[left_stance_transition], -1, dims=1
			)
			self.step_length_history[left_stance_transition, -1] = step_lengths_L[left_stance_transition]

			self.step_frequency_history[left_stance_transition] = torch.roll(
				self.step_frequency_history[left_stance_transition], -1, dims=1
			)
			self.step_frequency_history[left_stance_transition, -1] = step_freqs_L[left_stance_transition]

			self.last_step_time_lr[left_stance_transition, 0] = current_time[left_stance_transition]
			self.last_step_pos_lr[left_stance_transition, 0, :] = current_pos[left_stance_transition]

		if torch.any(right_stance_transition):
			step_lengths_R = torch.norm(current_pos - self.last_step_pos_lr[:, 1, :], dim=1)
			step_durations_R = current_time - self.last_step_time_lr[:, 1]
			step_freqs_R = torch.where(
				(step_durations_R > 0) & (self.last_step_time_lr[:, 1] > 0),
				1.0 / step_durations_R,
				torch.zeros_like(step_durations_R),
			)

			self.latest_step_length_lr[right_stance_transition, 1] = step_lengths_R[right_stance_transition]
			self.latest_step_frequency_lr[right_stance_transition, 1] = step_freqs_R[right_stance_transition]

			self.last_step_time_lr[right_stance_transition, 1] = current_time[right_stance_transition]
			self.last_step_pos_lr[right_stance_transition, 1, :] = current_pos[right_stance_transition]

	def _build_step_features(self):
		return {
			'step_length_history': self.step_length_history,
			'step_frequency_history': self.step_frequency_history,
			'step_length': self.step_length_history[:, -1],
			'step_frequency': self.step_frequency_history[:, -1],
		}

	def _combine_feature_scores(self, feature_scores, keys):
		tensors = [feature_scores[k] for k in keys if k in feature_scores]
		if not tensors:
			return torch.zeros(self.num_envs, device=self.device)
		return torch.mean(torch.stack(tensors, dim=0), dim=0)

	def _evaluate_human_like_reward(self):
		step_features = self._build_step_features()
		reward, details = self.human_like_gait_reward.evaluate(
			self.gait_history, self.pose_history, self.arm_history, step_features
		)
		reward = torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)
		feature_scores = details.get('feature_scores', {})

		components = {
			'gait_naturalness': self._combine_feature_scores(feature_scores, [
				'phase_regularity', 'velocity_consistency', 'movement_smoothness', 'direction_stability'
			]),
			'body_posture': self._combine_feature_scores(feature_scores, [
				'body_sway_balance'
			]),
			'arm_coordination': self._combine_feature_scores(feature_scores, [
				'arm_leg_phase_delta', 'arm_swing_balance', 'arm_smoothness'
			]),
			'arm_swing_phase': feature_scores.get('arm_leg_phase_delta', torch.zeros(self.num_envs, device=self.device)),
			'step_characteristics': self._combine_feature_scores(feature_scores, [
				'step_length_consistency', 'step_frequency_consistency'
			]),
			'movement_fluency': self._combine_feature_scores(feature_scores, [
				'movement_smoothness', 'energy_proxy'
			]),
			'human_similarity': details.get('human_similarity', torch.zeros(self.num_envs, device=self.device)),
			'feature_quality': details.get('feature_quality', torch.zeros(self.num_envs, device=self.device)),
			'human_like_total': reward,
		}

		self._style_reward_cache = reward
		self._style_component_cache = components
		self._style_detail_cache = details
		self._style_cache_valid = True
		self._update_style_stats(reward)

	def _update_style_stats(self, reward):
		if reward.numel() == 0:
			return
		robot = self.robot
		robot.extras['style_reward_mean'] = float(torch.mean(reward).item())
		robot.extras['style_reward_std'] = float(torch.std(reward, unbiased=False).item())
		robot.extras['style_reward_min'] = float(torch.min(reward).item())
		robot.extras['style_reward_max'] = float(torch.max(reward).item())
		outlier = torch.logical_or(torch.any(reward > 1.2), torch.any(reward < -0.2))
		robot.extras['style_reward_outlier'] = bool(outlier.item())

	def _ensure_style_cache(self):
		if not self._style_cache_valid:
			self._evaluate_human_like_reward()

	def get_component(self, name):
		self._ensure_style_cache()
		return self._style_component_cache.get(name, torch.zeros(self.num_envs, device=self.device))


	def reward_gait_naturalness(self):
		return self.get_component('gait_naturalness')

	def reward_body_posture(self):
		return self.get_component('body_posture')

	def reward_arm_coordination(self):
		return self.get_component('arm_coordination')

	def reward_arm_swing_phase(self):
		return self.get_component('arm_swing_phase')

	def reward_step_characteristics(self):
		return self.get_component('step_characteristics')

	def reward_movement_fluency(self):
		return self.get_component('movement_fluency')

	def reset(self, env_ids):
		if len(env_ids) == 0:
			return
		self.gait_history[env_ids] = 0
		self.pose_history[env_ids] = 0
		self.arm_history[env_ids] = 0
		self.step_length_history[env_ids] = 0
		self.step_frequency_history[env_ids] = 0
		self.latest_step_length_lr[env_ids] = 0
		self.latest_step_frequency_lr[env_ids] = 0
		self.last_step_time_lr[env_ids] = 0
		self.last_step_pos_lr[env_ids] = 0
		self._style_reward_cache[env_ids] = 0
		self._style_component_cache = {}
		self._style_detail_cache = {}
		self._style_cache_valid = False
