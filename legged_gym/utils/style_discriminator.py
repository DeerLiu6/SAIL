import isaacgym  # pylint: disable=unused-import
from typing import Dict, Optional, Tuple

import torch

from rsl_rl.modules import StyleDiscriminator

class HumanGaitAnalyzer:
    """Human gait analyzer - analyzes human walking characteristics."""
    
    def __init__(self, device='cuda'):
        self.device = device
        # Target human gait statistics: mean, std, and valid range.
        self.human_gait_features = {
            'step_length': {'mean': 0.6, 'std': 0.08, 'range': (0.4, 0.85)},
            'step_frequency': {'mean': 1.9, 'std': 0.25, 'range': (1.1, 2.6)},
            'arm_swing_amplitude': {'mean': 0.42, 'std': 0.08, 'range': (0.25, 0.65)},
            'body_sway': {'mean': 0.11, 'std': 0.02, 'range': (0.05, 0.18)},
            'forward_tilt': {'mean': 0.05, 'std': 0.02, 'range': (0.0, 0.1)},
            'lateral_drift': {'mean': 0.0, 'std': 0.04, 'range': (-0.15, 0.15)},
            'arm_phase_offset': {'mean': 0.5, 'std': 0.08, 'range': (0.35, 0.65)},
        }
        self.feature_weights = {
            'step_length': 1.0,
            'step_frequency': 1.0,
            'arm_swing_amplitude': 0.8,
            'body_sway': 0.8,
            'forward_tilt': 0.5,
            'lateral_drift': 0.7,
            'arm_phase_offset': 0.7,
        }
    
    def analyze_gait_similarity(self, robot_gait_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Map robot gait statistics to a human-prior similarity score."""
        num_envs = self._infer_batch(robot_gait_data)
        if num_envs <= 0:
            return torch.zeros(1, device=self.device)

        total = torch.zeros(num_envs, device=self.device)
        total_weight = 0.0

        for name, target in self.human_gait_features.items():
            value = robot_gait_data.get(name)
            if value is None:
                continue
            tensor = self._to_tensor(value, num_envs)
            score = self._gaussian_score(tensor, target['mean'], target['std'])
            if target.get('range') is not None:
                score = self._apply_range_penalty(score, tensor, target['range'])
            weight = self.feature_weights.get(name, 1.0)
            total += weight * score
            total_weight += weight

        total, total_weight = self._blend_combo_scores(total, total_weight, robot_gait_data, num_envs)
        if total_weight <= 0:
            return torch.zeros(num_envs, device=self.device)
        return total / total_weight

    def _infer_batch(self, data: Dict[str, torch.Tensor]) -> int:
        for value in data.values():
            if isinstance(value, torch.Tensor) and value.numel() > 0:
                return value.shape[0]
        return 0

    def _to_tensor(self, value, num_envs: int) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            tensor = value.to(self.device)
        else:
            tensor = torch.as_tensor(value, device=self.device, dtype=torch.float32)
        if tensor.ndim == 0:
            tensor = tensor.repeat(num_envs)
        elif tensor.shape[0] != num_envs:
            if tensor.numel() == 1:
                tensor = tensor.repeat(num_envs)
            elif tensor.ndim == 1:
                tensor = tensor[:1].repeat(num_envs)
            else:
                tensor = tensor.reshape(1, -1).repeat(num_envs, 1)
        return tensor

    @staticmethod
    def _gaussian_score(value: torch.Tensor, mean: float, std: float) -> torch.Tensor:
        std = max(std, 1e-6)
        return torch.exp(-0.5 * torch.square((value - mean) / std))

    @staticmethod
    def _apply_range_penalty(score: torch.Tensor, value: torch.Tensor, valid_range: Tuple[float, float]) -> torch.Tensor:
        low, high = valid_range
        inside = (value >= low) & (value <= high)
        return torch.where(inside, score, 0.4 * score)

    def _blend_combo_scores(
        self,
        current: torch.Tensor,
        total_weight: float,
        data: Dict[str, torch.Tensor],
        num_envs: int,
    ) -> Tuple[torch.Tensor, float]:
        if 'step_length' in data and 'step_frequency' in data:
            step_speed = self._to_tensor(data['step_length'], num_envs) * self._to_tensor(data['step_frequency'], num_envs)
            target_speed = self.human_gait_features['step_length']['mean'] * self.human_gait_features['step_frequency']['mean']
            speed_score = self._gaussian_score(step_speed, target_speed, 0.25)
            current += 0.7 * speed_score
            total_weight += 0.7

        if 'arm_swing_amplitude' in data and 'arm_phase_offset' in data:
            amp = self._to_tensor(data['arm_swing_amplitude'], num_envs)
            phase = self._to_tensor(data['arm_phase_offset'], num_envs)
            arm_combo = self._gaussian_score(amp, 0.42, 0.08) * self._gaussian_score(phase, 0.5, 0.08)
            current += 0.5 * arm_combo
            total_weight += 0.5

        if 'lateral_drift' in data:
            drift = self._to_tensor(data['lateral_drift'], num_envs)
            drift_score = self._gaussian_score(drift, 0.0, 0.04)
            current += 0.3 * drift_score
            total_weight += 0.3

        return current, total_weight

"""The StyleDiscriminator network is defined in rsl_rl.modules.
This file only handles feature engineering and reward wrapping to keep the RL stack and discriminator saved together."""

class GaitFeatureExtractor:
    """Gait feature extractor."""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.feature_order = [
            'phase_regularity',
            'velocity_consistency',
            'height_oscillation_score',
            'body_sway_balance',
            'arm_leg_phase_delta',
            'arm_swing_balance',
            'arm_smoothness',
            'step_length_consistency',
            'step_frequency_consistency',
            'movement_smoothness',
            'direction_stability',
            'energy_proxy',
        ]
        self.feature_weights = [1.0, 1.0, 0.0, 0.6, 0.8, 0.6, 0.5, 0.7, 0.7, 0.8, 0.6, 0.4]
        self.feature_dim = len(self.feature_order)
        self._eps = 1e-6
    
    def extract_gait_features(
        self,
        gait_history: torch.Tensor,
        pose_history: torch.Tensor,
        arm_history: torch.Tensor,
        step_features: Dict,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Extract gait feature vectors and interpretable statistics."""
        num_envs = gait_history.shape[0]
        if num_envs == 0:
            return torch.zeros(0, self.feature_dim, device=self.device), {}

        gait_window = self._tail(gait_history, 6)
        pose_window = self._tail(pose_history, 5)
        arm_window = self._tail(arm_history, 6)

        feature_map = {
            'phase_regularity': self._compute_phase_regularity(gait_window[:, :, 4]),
            'velocity_consistency': self._compute_velocity_consistency(gait_window[:, :, 2:4]),
            'height_oscillation_score': self._compute_height_oscillation_score(gait_window[:, :, 5]),
            'body_sway_balance': self._compute_body_sway_balance(pose_window[:, :, 0]),
            'arm_leg_phase_delta': self._compute_phase_delta(arm_window[:, :, 2], arm_window[:, :, 3]),
            'arm_swing_balance': self._compute_arm_balance(arm_window[:, :, :2]),
            'arm_smoothness': self._compute_arm_smoothness(arm_window[:, :, :2]),
            'movement_smoothness': self._compute_movement_smoothness(gait_window[:, :, 2:4]),
            'direction_stability': self._compute_direction_stability(gait_window[:, :, 2:4]),
            'energy_proxy': self._compute_energy_proxy(gait_window[:, :, 2:4]),
        }

        step_length_series = self._resolve_step_history(step_features, 'step_length', num_envs)
        step_frequency_series = self._resolve_step_history(step_features, 'step_frequency', num_envs)
        feature_map['step_length_consistency'] = self._history_consistency(step_length_series, 0.04)
        feature_map['step_frequency_consistency'] = self._history_consistency(step_frequency_series, 0.2)

        interpretable = {
            'step_length': self._resolve_step_scalar(step_features, 'step_length', num_envs),
            'step_frequency': self._resolve_step_scalar(step_features, 'step_frequency', num_envs),
            'arm_swing_amplitude': self._arm_amplitude(arm_window[:, :, :2]),
            'body_sway': self._peak_to_peak(pose_window[:, :, 0]),
            'vertical_bounce': self._peak_to_peak(gait_window[:, :, 5]),
            'forward_tilt': pose_window[:, -1, 1],
            'lateral_drift': gait_window[:, -1, 1] - gait_window[:, 0, 1],
            'arm_phase_offset': self._mean_phase_offset(arm_window[:, :, 2], arm_window[:, :, 3]),
        }

        feature_tensor = torch.zeros(num_envs, self.feature_dim, device=self.device)
        for idx, name in enumerate(self.feature_order):
            feature_tensor[:, idx] = feature_map.get(name, torch.zeros(num_envs, device=self.device))

        return feature_tensor, interpretable

    def _tail(self, tensor: torch.Tensor, length: int) -> torch.Tensor:
        if tensor.shape[1] <= length:
            return tensor
        return tensor[:, -length:, ...]

    def _finite_diff(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor[:, 1:, ...] - tensor[:, :-1, ...]

    def _peak_to_peak(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.shape[1] == 0:
            return torch.zeros(tensor.shape[0], device=self.device)
        return torch.max(tensor, dim=1).values - torch.min(tensor, dim=1).values

    def _compute_phase_regularity(self, phase_history: torch.Tensor) -> torch.Tensor:
        if phase_history.shape[1] < 2:
            return torch.zeros(phase_history.shape[0], device=self.device)
        deltas = torch.remainder(self._finite_diff(phase_history), 1.0)
        return torch.exp(-8.0 * torch.std(deltas, dim=1))

    def _compute_velocity_consistency(self, velocity_history: torch.Tensor) -> torch.Tensor:
        if velocity_history.shape[1] < 2:
            return torch.zeros(velocity_history.shape[0], device=self.device)
        speeds = torch.norm(velocity_history, dim=2)
        return torch.exp(-torch.std(speeds, dim=1) / 0.25)

    def _compute_height_oscillation_score(self, height_history: torch.Tensor) -> torch.Tensor:
        amplitude = self._peak_to_peak(height_history)
        return torch.exp(-torch.square((amplitude - 0.055) / 0.02))

    def _compute_body_sway_balance(self, roll_history: torch.Tensor) -> torch.Tensor:
        amplitude = self._peak_to_peak(roll_history)
        return torch.exp(-torch.square((amplitude - 0.11) / 0.03))

    def _compute_phase_delta(self, left_phase: torch.Tensor, right_phase: torch.Tensor) -> torch.Tensor:
        if left_phase.shape[1] < 2 or right_phase.shape[1] < 2:
            return torch.zeros(left_phase.shape[0], device=self.device)
        phase_gap = torch.abs(left_phase - right_phase)
        wrapped_gap = torch.minimum(phase_gap, 1.0 - phase_gap)
        delta_score = torch.exp(-torch.square((wrapped_gap - 0.5) / 0.08))
        return torch.mean(delta_score, dim=1)

    def _mean_phase_offset(self, left_phase: torch.Tensor, right_phase: torch.Tensor) -> torch.Tensor:
        if left_phase.shape[1] == 0:
            return torch.zeros(left_phase.shape[0], device=self.device)
        phase_gap = torch.abs(left_phase - right_phase)
        wrapped_gap = torch.minimum(phase_gap, 1.0 - phase_gap)
        return torch.mean(wrapped_gap, dim=1)

    def _compute_arm_balance(self, arm_series: torch.Tensor) -> torch.Tensor:
        if arm_series.shape[1] < 2:
            return torch.zeros(arm_series.shape[0], device=self.device)
        left = arm_series[:, :, 0]
        right = arm_series[:, :, 1]
        left_amp = self._peak_to_peak(left.unsqueeze(-1)).squeeze(-1)
        right_amp = self._peak_to_peak(right.unsqueeze(-1)).squeeze(-1)
        total = left_amp + right_amp + self._eps
        balance = 1.0 - torch.abs(left_amp - right_amp) / total
        return torch.clamp(balance, 0.0, 1.0)

    def _arm_amplitude(self, arm_series: torch.Tensor) -> torch.Tensor:
        left_amp = self._peak_to_peak(arm_series[:, :, 0].unsqueeze(-1)).squeeze(-1)
        right_amp = self._peak_to_peak(arm_series[:, :, 1].unsqueeze(-1)).squeeze(-1)
        return 0.5 * (left_amp + right_amp)

    def _compute_arm_smoothness(self, arm_series: torch.Tensor) -> torch.Tensor:
        if arm_series.shape[1] < 2:
            return torch.zeros(arm_series.shape[0], device=self.device)
        diff = self._finite_diff(arm_series)
        return torch.exp(-torch.mean(torch.square(diff), dim=(1, 2)) / 0.02)

    def _compute_movement_smoothness(self, velocity_history: torch.Tensor) -> torch.Tensor:
        if velocity_history.shape[1] < 2:
            return torch.zeros(velocity_history.shape[0], device=self.device)
        accel = self._finite_diff(velocity_history)
        accel_norm = torch.norm(accel, dim=2)
        return torch.exp(-torch.mean(accel_norm, dim=1) / 0.6)

    def _compute_direction_stability(self, velocity_history: torch.Tensor) -> torch.Tensor:
        if velocity_history.shape[1] < 2:
            return torch.zeros(velocity_history.shape[0], device=self.device)
        headings = torch.atan2(velocity_history[:, :, 1], velocity_history[:, :, 0])
        return torch.exp(-torch.std(headings, dim=1) / 0.4)

    def _compute_energy_proxy(self, velocity_history: torch.Tensor) -> torch.Tensor:
        speeds = torch.norm(velocity_history, dim=2)
        target_speed = 1.1
        return torch.exp(-torch.square((torch.mean(speeds, dim=1) - target_speed) / 0.4))

    def _history_consistency(self, series: torch.Tensor, tolerance: float) -> torch.Tensor:
        if series.shape[1] < 2:
            return torch.zeros(series.shape[0], device=self.device)
        variance = torch.var(series, dim=1, unbiased=False)
        return torch.exp(-variance / (tolerance + self._eps))

    def _resolve_step_history(self, step_features: Dict, key: str, num_envs: int) -> torch.Tensor:
        hist_key = f'{key}_history'
        if hist_key in step_features:
            tensor = self._to_tensor(step_features[hist_key], num_envs)
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(1)
            return tensor
        scalar = self._resolve_step_scalar(step_features, key, num_envs)
        return scalar.unsqueeze(1)

    def _resolve_step_scalar(self, step_features: Dict, key: str, num_envs: int) -> torch.Tensor:
        if key in step_features:
            tensor = self._to_tensor(step_features[key], num_envs)
            if tensor.ndim > 1:
                tensor = tensor[:, -1]
        else:
            hist_key = f'{key}_history'
            tensor = self._to_tensor(step_features.get(hist_key, torch.zeros(num_envs, 1, device=self.device)), num_envs)
            if tensor.ndim > 1:
                tensor = tensor[:, -1]
        return tensor

    def _to_tensor(self, value, num_envs: int) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            tensor = value.to(self.device)
        else:
            tensor = torch.as_tensor(value, device=self.device, dtype=torch.float32)
        if tensor.ndim == 0:
            tensor = tensor.repeat(num_envs)
        elif tensor.shape[0] != num_envs:
            if tensor.numel() == 1:
                tensor = tensor.repeat(num_envs)
            elif tensor.ndim == 1:
                tensor = tensor[:1].repeat(num_envs)
            else:
                tensor = tensor.reshape(1, -1).repeat(num_envs, 1)
        return tensor

class HumanLikeGaitReward:
    """Human-like gait reward calculator."""
    
    def __init__(self, device='cuda', use_discriminator: bool = False, discriminator_cfg: Optional[Dict] = None):
        self.device = device
        self.gait_analyzer = HumanGaitAnalyzer(device)
        self.feature_extractor = GaitFeatureExtractor(device)
        self.discriminator = None
        self.use_discriminator = use_discriminator
        if use_discriminator:
            cfg = discriminator_cfg or {}
            hidden_dims = cfg.get('hidden_dims')
            dropout_p = cfg.get('dropout_p', 0.1)
            self.discriminator = StyleDiscriminator(
                input_dim=self.feature_extractor.feature_dim,
                hidden_dims=hidden_dims,
                dropout_p=dropout_p,
                device=device,
            )
            self.discriminator.eval()
    
    def compute_human_like_reward(self, gait_history: torch.Tensor, pose_history: torch.Tensor,
                                arm_history: torch.Tensor, step_features: Dict) -> torch.Tensor:
        """Compute the human-like gait reward.

        Args:
            gait_history: [num_envs, T_g, 6] gait buffer with at least position, velocity, phase, and height.
            pose_history: [num_envs, T_p, 4] pose buffer with roll/pitch/yaw/base_height.
            arm_history: [num_envs, T_a, 4] arm buffer with left/right shoulder angles and corresponding leg phases.
            step_features: stats for the discriminator; keys include step_length, step_frequency and *_history.
        """
        reward, _ = self.evaluate(gait_history, pose_history, arm_history, step_features)
        return reward

    def evaluate(self, gait_history: torch.Tensor, pose_history: torch.Tensor,
                 arm_history: torch.Tensor, step_features: Dict) -> Tuple[torch.Tensor, Dict]:
        """Return the reward and intermediate features for diagnostics/visualization."""
        with torch.no_grad():
            feature_tensor, interpretable = self.feature_extractor.extract_gait_features(
                gait_history, pose_history, arm_history, step_features
            )

        feature_scores = {
            name: feature_tensor[:, idx]
            for idx, name in enumerate(self.feature_extractor.feature_order)
        } if feature_tensor.shape[1] == self.feature_extractor.feature_dim else {}

        human_similarity = self.gait_analyzer.analyze_gait_similarity(interpretable)
        feature_quality = self._score_feature_quality(feature_tensor)

        reward = 0.55 * human_similarity + 0.45 * feature_quality
        details = {
            'human_similarity': human_similarity,
            'feature_quality': feature_quality,
            'feature_scores': feature_scores,
            'interpretable': interpretable,
        }

        if self.discriminator is not None and feature_tensor.shape[1] == self.feature_extractor.feature_dim:
            disc_score = self.discriminator(feature_tensor).squeeze(-1)
            details['discriminator_score'] = disc_score
            reward = 0.5 * reward + 0.5 * disc_score

        details['reward'] = reward
        return reward, details

    def _score_feature_quality(self, feature_tensor: torch.Tensor) -> torch.Tensor:
        if feature_tensor.numel() == 0:
            return torch.zeros(feature_tensor.shape[0], device=self.device)
        weights = torch.tensor(self.feature_extractor.feature_weights, device=feature_tensor.device)
        weights = weights / (torch.sum(weights) + 1e-6)
        return torch.sum(feature_tensor * weights, dim=1)

    def load_discriminator_state(self, state_dict: Dict):
        if self.discriminator is None:
            raise RuntimeError('Style discriminator is disabled. Instantiate with use_discriminator=True.')
        self.discriminator.load_state_dict(state_dict)
        self.discriminator.eval()
    
    def get_gait_analysis_report(self, gait_history: torch.Tensor, pose_history: torch.Tensor,
                               arm_history: torch.Tensor, step_features: Dict) -> Dict:
        """Get a gait analysis report."""
        with torch.no_grad():
            feature_tensor, interpretable = self.feature_extractor.extract_gait_features(
                gait_history, pose_history, arm_history, step_features
            )

        report = {}
        if feature_tensor.shape[0] > 0:
            for idx, name in enumerate(self.feature_extractor.feature_order):
                report[name] = float(feature_tensor[:, idx].mean().item())
        else:
            for name in self.feature_extractor.feature_order:
                report[name] = 0.0

        for key, value in interpretable.items():
            if value.numel() == 0:
                report[key] = 0.0
            else:
                report[key] = float(value.mean().item())
        return report
