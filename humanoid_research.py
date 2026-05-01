import argparse
import csv
import json
import os
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple, Type

import gymnasium as gym
import imageio
import matplotlib
import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.sac.policies import Actor as SB3Actor
from stable_baselines3.sac.policies import SACPolicy

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT_IDX = torch.tensor(list(range(0, 5)) + list(range(22, 28)), dtype=torch.long)
KIN_IDX = torch.tensor(
    list(range(5, 22)) + list(range(28, 45)) + list(range(269, 292)),
    dtype=torch.long,
)
CONTACT_IDX = torch.tensor(
    list(range(45, 185)) + list(range(185, 269)) + list(range(292, 376)),
    dtype=torch.long,
)
KIN_ROOT_IDX = torch.tensor(sorted(set(ROOT_IDX.tolist() + KIN_IDX.tolist())), dtype=torch.long)
LOG_STD_MAX, LOG_STD_MIN = 2, -20


def ram_gb() -> float:
    return psutil.virtual_memory().used / 1e9


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def make_env(env_id: str, seed: int):
    def _init():
        env = gym.make(env_id)
        env = Monitor(env)
        env.reset(seed=seed)
        return env

    return _init


def build_vec_env(config: Dict) -> VecNormalize:
    base_env = DummyVecEnv([make_env(config["env_id"], config["seed"])])
    return VecNormalize(
        base_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=config["clip_obs"],
        clip_reward=config["clip_reward"],
        gamma=config["gamma"],
    )


def make_render_env(env_id: str, seed: int):
    env = gym.make(env_id, render_mode="rgb_array")
    env.reset(seed=seed)
    return env


def interpolate_metric_at_step(steps: List[int], values: List[float], target_step: int) -> float | None:
    if not steps:
        return None
    pairs = [(s, v) for s, v in zip(steps, values) if np.isfinite(v)]
    if not pairs:
        return None
    steps_arr = np.array([p[0] for p in pairs], dtype=float)
    vals_arr = np.array([p[1] for p in pairs], dtype=float)
    if target_step <= steps_arr[0]:
        return float(vals_arr[0])
    if target_step > steps_arr[-1]:
        return None
    return float(np.interp(target_step, steps_arr, vals_arr))


class KinematicsActor(SB3Actor):
    def __init__(
        self,
        observation_space,
        action_space,
        net_arch: List[int],
        features_extractor,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        **kwargs,
    ):
        super().__init__(
            observation_space,
            action_space,
            net_arch,
            features_extractor,
            features_dim,
            activation_fn,
            **kwargs,
        )
        kin_dim = len(KIN_IDX)
        act_dim = action_space.shape[0]
        layers, prev = [], kin_dim
        for hidden in net_arch:
            layers += [nn.Linear(prev, hidden), activation_fn()]
            prev = hidden
        self.latent_pi = nn.Sequential(*layers)
        self.mu = nn.Linear(prev, act_dim)
        self.log_std = nn.Linear(prev, act_dim)

    def get_action_dist_params(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        kin = obs[:, KIN_IDX.to(obs.device)]
        latent = self.latent_pi(kin)
        mean = self.mu(latent)
        log_std = self.log_std(latent).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std, {}


class KinematicsRootActor(SB3Actor):
    def __init__(
        self,
        observation_space,
        action_space,
        net_arch: List[int],
        features_extractor,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        **kwargs,
    ):
        super().__init__(
            observation_space,
            action_space,
            net_arch,
            features_extractor,
            features_dim,
            activation_fn,
            **kwargs,
        )
        input_dim = len(KIN_ROOT_IDX)
        act_dim = action_space.shape[0]
        layers, prev = [], input_dim
        for hidden in net_arch:
            layers += [nn.Linear(prev, hidden), activation_fn()]
            prev = hidden
        self.latent_pi = nn.Sequential(*layers)
        self.mu = nn.Linear(prev, act_dim)
        self.log_std = nn.Linear(prev, act_dim)

    def get_action_dist_params(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        kin_root = obs[:, KIN_ROOT_IDX.to(obs.device)]
        latent = self.latent_pi(kin_root)
        mean = self.mu(latent)
        log_std = self.log_std(latent).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std, {}


class ContactAttentionWeighter(nn.Module):
    def __init__(
        self,
        root_dim: int = 11,
        contact_dim: int = 308,
        embed_dim: int = 64,
        num_heads: int = 4,
        contact_hidden: int = 256,
        weight_mode: str = "sigmoid2x",
        attn_scale: float = 0.25,
    ):
        super().__init__()
        self.weight_mode = weight_mode
        self.attn_scale = attn_scale
        self.contact_encoder = nn.Sequential(
            nn.Linear(contact_dim, contact_hidden),
            nn.ReLU(),
            nn.Linear(contact_hidden, embed_dim),
            nn.ReLU(),
        )
        self.root_proj = nn.Sequential(nn.Linear(root_dim, embed_dim), nn.ReLU())
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.0,
        )
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.weight_head = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        device = obs.device
        root = obs[:, ROOT_IDX.to(device)]
        contact = obs[:, CONTACT_IDX.to(device)]
        root_emb = self.root_proj(root)
        contact_emb = self.contact_encoder(contact)
        attended, _ = self.cross_attn(
            root_emb.unsqueeze(1),
            contact_emb.unsqueeze(1),
            contact_emb.unsqueeze(1),
        )
        attended = self.layer_norm(attended.squeeze(1) + root_emb)
        raw_weight = self.weight_head(attended)
        if self.weight_mode == "centered_tanh":
            centered = raw_weight - raw_weight.mean(dim=0, keepdim=True)
            return 1.0 + self.attn_scale * torch.tanh(centered)
        if self.weight_mode == "normalized_exp":
            centered = raw_weight - raw_weight.mean(dim=0, keepdim=True)
            positive = torch.exp(self.attn_scale * torch.tanh(centered))
            return positive / positive.mean(dim=0, keepdim=True).detach().clamp_min(1e-6)
        return 2.0 * torch.sigmoid(raw_weight)


class KinematicsOnlySACPolicy(SACPolicy):
    def make_actor(self, features_extractor=None) -> KinematicsActor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return KinematicsActor(**actor_kwargs).to(self.device)


class AttentiveSACPolicy(SACPolicy):
    def __init__(
        self,
        *args,
        embed_dim=64,
        num_heads=4,
        contact_hidden=256,
        weight_mode="sigmoid2x",
        attn_scale=0.25,
        **kwargs,
    ):
        self._embed_dim = embed_dim
        self._num_heads = num_heads
        self._contact_hidden = contact_hidden
        self._weight_mode = weight_mode
        self._attn_scale = attn_scale
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule) -> None:
        super()._build(lr_schedule)
        self.attn_weighter = ContactAttentionWeighter(
            embed_dim=self._embed_dim,
            num_heads=self._num_heads,
            contact_hidden=self._contact_hidden,
            weight_mode=self._weight_mode,
            attn_scale=self._attn_scale,
        ).to(self.device)
        self.actor.optimizer = self.optimizer_class(
            list(self.actor.parameters()) + list(self.attn_weighter.parameters()),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

    def make_actor(self, features_extractor=None) -> KinematicsActor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return KinematicsActor(**actor_kwargs).to(self.device)


class FullObsAttentionLossPolicy(SACPolicy):
    def __init__(
        self,
        *args,
        embed_dim=64,
        num_heads=4,
        contact_hidden=256,
        weight_mode="sigmoid2x",
        attn_scale=0.25,
        **kwargs,
    ):
        self._embed_dim = embed_dim
        self._num_heads = num_heads
        self._contact_hidden = contact_hidden
        self._weight_mode = weight_mode
        self._attn_scale = attn_scale
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule) -> None:
        super()._build(lr_schedule)
        self.attn_weighter = ContactAttentionWeighter(
            embed_dim=self._embed_dim,
            num_heads=self._num_heads,
            contact_hidden=self._contact_hidden,
            weight_mode=self._weight_mode,
            attn_scale=self._attn_scale,
        ).to(self.device)
        self.actor.optimizer = self.optimizer_class(
            list(self.actor.parameters()) + list(self.attn_weighter.parameters()),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )


class KinRootAttentiveSACPolicy(AttentiveSACPolicy):
    def make_actor(self, features_extractor=None) -> KinematicsRootActor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return KinematicsRootActor(**actor_kwargs).to(self.device)


class CenteredAttentiveSACPolicy(AttentiveSACPolicy):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("weight_mode", "centered_tanh")
        kwargs.setdefault("attn_scale", 0.25)
        super().__init__(*args, **kwargs)


class NormalizedAttentiveSACPolicy(AttentiveSACPolicy):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("weight_mode", "normalized_exp")
        kwargs.setdefault("attn_scale", 0.50)
        super().__init__(*args, **kwargs)


class KinRootNormalizedAttentiveSACPolicy(NormalizedAttentiveSACPolicy):
    def make_actor(self, features_extractor=None) -> KinematicsRootActor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return KinematicsRootActor(**actor_kwargs).to(self.device)


class AttentionLossMixin:
    def _get_log_ent_coef(self):
        for attr in ("log_ent_coef", "log_ent_coef_param"):
            if hasattr(self, attr):
                return getattr(self, attr)
        params = self.ent_coef_optimizer.param_groups[0].get("params", [])
        return params[0] if params else None

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(
            [self.actor.optimizer, self.critic.optimizer, self.ent_coef_optimizer]
        )
        grad_clip = getattr(self, "max_grad_norm", 10.0)

        actor_losses, critic_losses, ent_coef_losses = [], [], []
        ent_coefs, mean_qs, attn_means, attn_stds = [], [], [], []
        log_ent_coef = self._get_log_ent_coef()

        for _ in range(gradient_steps):
            self._n_updates += 1
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            obs = replay_data.observations

            with torch.no_grad():
                _, log_prob = self.actor.action_log_prob(obs)
                log_prob = log_prob.reshape(-1, 1)
            ent_coef = torch.exp(log_ent_coef.detach())
            ent_coef_loss = -(log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
            self.ent_coef_optimizer.zero_grad()
            ent_coef_loss.backward()
            self.ent_coef_optimizer.step()
            ent_coef_losses.append(ent_coef_loss.item())
            ent_coefs.append(ent_coef.item())

            with torch.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                next_q_values = torch.cat(
                    self.critic_target(replay_data.next_observations, next_actions), dim=1
                )
                next_q, _ = torch.min(next_q_values, dim=1, keepdim=True)
                target_q = replay_data.rewards + (1 - replay_data.dones) * self.gamma * (
                    next_q - ent_coef * next_log_prob.reshape(-1, 1)
                )

            current_q = self.critic(obs, replay_data.actions)
            critic_loss = sum(F.mse_loss(q, target_q) for q in current_q)
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), grad_clip)
            self.critic.optimizer.step()
            critic_losses.append(critic_loss.item())

            for param in self.critic.parameters():
                param.requires_grad_(False)

            actions_pi, log_prob = self.actor.action_log_prob(obs)
            log_prob = log_prob.reshape(-1, 1)
            q_pi = torch.cat(self.critic(obs, actions_pi), dim=1)
            min_q, _ = torch.min(q_pi, dim=1, keepdim=True)
            attention_weights = self.policy.attn_weighter(obs)
            per_sample_loss = ent_coef * log_prob - min_q
            actor_loss = (attention_weights * per_sample_loss).mean()

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.actor.parameters()) + list(self.policy.attn_weighter.parameters()),
                grad_clip,
            )
            self.actor.optimizer.step()

            actor_losses.append(actor_loss.item())
            mean_qs.append(min_q.mean().item())
            attn_means.append(attention_weights.mean().item())
            attn_stds.append(attention_weights.std(unbiased=False).item())

            for param in self.critic.parameters():
                param.requires_grad_(True)

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.mul_(1 - self.tau).add_(self.tau * param.data)

        self.logger.record("train/n_updates", self._n_updates)
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
        self.logger.record("train/mean_q1", np.mean(mean_qs))
        self.logger.record("train/attn_mean", np.mean(attn_means))
        self.logger.record("train/attn_std", np.mean(attn_stds))


class AttentiveSAC(AttentionLossMixin, SAC):
    pass


class FullObsAttentionLossSAC(AttentionLossMixin, SAC):
    pass


def get_log_alpha(model) -> torch.Tensor | None:
    if not hasattr(model, "ent_coef_optimizer"):
        return None
    params = model.ent_coef_optimizer.param_groups[0].get("params", [])
    return params[0] if params else None


class MetricsCallback(BaseCallback):
    def __init__(
        self,
        run_dir: Path,
        log_interval: int = 2000,
        reward_threshold: float = 3000.0,
        checkpoint_freq: int = 25000,
        best_model_window: int = 50,
        alpha_min: float = 0.05,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.run_dir = run_dir
        self.log_interval = log_interval
        self.reward_threshold = reward_threshold
        self.checkpoint_freq = checkpoint_freq
        self.best_model_window = best_model_window
        self.alpha_min = alpha_min
        self.log_alpha_min = float(np.log(alpha_min))

        self.timesteps: List[int] = []
        self.mean_rewards: List[float] = []
        self.std_rewards: List[float] = []
        self.min_rewards: List[float] = []
        self.max_rewards: List[float] = []
        self.ep_lengths: List[float] = []
        self.actor_losses: List[float] = []
        self.critic_losses: List[float] = []
        self.q_values: List[float] = []
        self.ent_coefs: List[float] = []
        self.fps_history: List[float] = []
        self.success_rates: List[float] = []
        self.ram_history: List[float] = []
        self.attn_means: List[float] = []
        self.attn_stds: List[float] = []

        self.ep_reward_window = deque(maxlen=100)
        self.ep_length_window = deque(maxlen=100)
        self.success_window = deque(maxlen=100)
        self.best_mean_reward = -np.inf
        self.best_step = 0
        self.last_time = time.time()
        self.last_steps = 0
        self.log_alpha_param = None

        self.metrics_path = self.run_dir / "metrics.csv"
        self.checkpoints_dir = ensure_dir(self.run_dir / "checkpoints")
        self.write_metrics_header()

    def write_metrics_header(self) -> None:
        with self.metrics_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "step",
                    "mean_reward",
                    "std_reward",
                    "min_reward",
                    "max_reward",
                    "ep_length",
                    "critic_loss",
                    "actor_loss",
                    "q_value",
                    "alpha",
                    "fps",
                    "ram_gb",
                    "success_rate",
                    "attn_mean",
                    "attn_std",
                ]
            )

    def _on_training_start(self) -> None:
        self.log_alpha_param = get_log_alpha(self.model)

    def _on_step(self) -> bool:
        if self.log_alpha_param is not None:
            with torch.no_grad():
                self.log_alpha_param.clamp_(min=self.log_alpha_min)

        for info in self.locals.get("infos", []):
            if "episode" in info:
                reward = info["episode"]["r"]
                length = info["episode"]["l"]
                self.ep_reward_window.append(reward)
                self.ep_length_window.append(length)
                self.success_window.append(int(reward > self.reward_threshold))

        if self.n_calls > 0 and self.n_calls % self.checkpoint_freq == 0:
            checkpoint_base = self.checkpoints_dir / f"model_{self.num_timesteps}"
            self.model.save(str(checkpoint_base))
            vec_env = self.model.get_vec_normalize_env()
            if vec_env is not None:
                vec_env.save(str(checkpoint_base) + "_vecnorm.pkl")

        if self.n_calls % self.log_interval != 0 or not self.ep_reward_window:
            return True

        logs = self.model.logger.name_to_value
        mean_reward = float(np.mean(self.ep_reward_window))
        std_reward = float(np.std(self.ep_reward_window))
        min_reward = float(np.min(self.ep_reward_window))
        max_reward = float(np.max(self.ep_reward_window))
        ep_length = float(np.mean(self.ep_length_window))
        success_rate = float(np.mean(self.success_window) * 100)
        actor_loss = float(logs.get("train/actor_loss", np.nan))
        critic_loss = float(logs.get("train/critic_loss", np.nan))
        q_value = float(
            logs.get("train/mean_q1", logs.get("train/q1_values", logs.get("train/qf1_values", np.nan)))
        )
        alpha = float(logs.get("train/ent_coef", np.nan))
        attn_mean = float(logs.get("train/attn_mean", np.nan))
        attn_std = float(logs.get("train/attn_std", np.nan))
        now = time.time()
        fps = (self.num_timesteps - self.last_steps) / max(now - self.last_time, 1e-6)
        self.last_time = now
        self.last_steps = self.num_timesteps
        used_ram = ram_gb()

        self.timesteps.append(self.num_timesteps)
        self.mean_rewards.append(mean_reward)
        self.std_rewards.append(std_reward)
        self.min_rewards.append(min_reward)
        self.max_rewards.append(max_reward)
        self.ep_lengths.append(ep_length)
        self.actor_losses.append(actor_loss)
        self.critic_losses.append(critic_loss)
        self.q_values.append(q_value)
        self.ent_coefs.append(alpha)
        self.fps_history.append(fps)
        self.success_rates.append(success_rate)
        self.ram_history.append(used_ram)
        self.attn_means.append(attn_mean)
        self.attn_stds.append(attn_std)

        with self.metrics_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    self.num_timesteps,
                    mean_reward,
                    std_reward,
                    min_reward,
                    max_reward,
                    ep_length,
                    critic_loss,
                    actor_loss,
                    q_value,
                    alpha,
                    fps,
                    used_ram,
                    success_rate,
                    attn_mean,
                    attn_std,
                ]
            )

        recent = list(self.ep_reward_window)[-self.best_model_window :]
        rolling_reward = float(np.mean(recent)) if recent else mean_reward
        if rolling_reward > self.best_mean_reward:
            self.best_mean_reward = rolling_reward
            self.best_step = self.num_timesteps
            self.model.save(str(self.run_dir / "best_model"))
            vec_env = self.model.get_vec_normalize_env()
            if vec_env is not None:
                vec_env.save(str(self.run_dir / "best_model_vecnorm.pkl"))

        if self.verbose:
            print(
                f"step={self.num_timesteps:>8,d} "
                f"reward={mean_reward:>8.1f} "
                f"best={self.best_mean_reward:>8.1f} "
                f"fps={fps:>6.1f} "
                f"alpha={alpha:>7.4f} "
                f"ram={used_ram:>5.1f}GB"
            )
        return True


def record_random_video(config: Dict, output_path: Path, n_episodes: int = 2, max_steps_per_ep: int = 1000) -> Dict:
    env = make_render_env(config["env_id"], config["seed"])
    frames = []
    rewards = []
    lengths = []
    try:
        for episode_idx in range(n_episodes):
            obs, _ = env.reset(seed=config["seed"] + episode_idx)
            done = False
            ep_reward = 0.0
            ep_length = 0
            while not done and ep_length < max_steps_per_ep:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, _ = env.step(action)
                frames.append(env.render())
                ep_reward += reward
                ep_length += 1
                done = terminated or truncated
            rewards.append(ep_reward)
            lengths.append(ep_length)
    finally:
        env.close()

    imageio.mimsave(output_path, frames, fps=config["video_fps"])
    return {
        "video_type": "random_before_training",
        "episodes": n_episodes,
        "mean_reward": float(np.mean(rewards)) if rewards else None,
        "mean_length": float(np.mean(lengths)) if lengths else None,
        "frames": len(frames),
        "path": str(output_path),
    }


def record_trained_video(
    model,
    vecnorm_path: Path,
    config: Dict,
    output_path: Path,
    deterministic: bool = True,
    n_episodes: int = 2,
    max_steps_per_ep: int = 1000,
) -> Dict:
    eval_base = DummyVecEnv([make_env(config["env_id"], config["seed"])])
    eval_env = VecNormalize.load(str(vecnorm_path), eval_base)
    eval_env.training = False
    eval_env.norm_reward = False
    render_env = make_render_env(config["env_id"], config["seed"])
    frames = []
    rewards = []
    lengths = []
    try:
        for episode_idx in range(n_episodes):
            obs = eval_env.reset()
            render_env.reset(seed=config["seed"] + episode_idx)
            done = False
            ep_reward = 0.0
            ep_length = 0
            while not done and ep_length < max_steps_per_ep:
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, _, done_arr, _ = eval_env.step(action)
                _, reward_raw, terminated, truncated, _ = render_env.step(action[0])
                frames.append(render_env.render())
                ep_reward += reward_raw
                ep_length += 1
                done = bool(done_arr[0]) or terminated or truncated
            rewards.append(ep_reward)
            lengths.append(ep_length)
    finally:
        eval_env.close()
        render_env.close()

    imageio.mimsave(output_path, frames, fps=config["video_fps"])
    return {
        "video_type": "trained_after_training",
        "episodes": n_episodes,
        "mean_reward": float(np.mean(rewards)) if rewards else None,
        "mean_length": float(np.mean(lengths)) if lengths else None,
        "frames": len(frames),
        "path": str(output_path),
    }


def build_model(experiment: str, train_env: VecNormalize, config: Dict, device: str):
    common_kwargs = dict(
        env=train_env,
        learning_rate=config["learning_rate"],
        buffer_size=config["buffer_size"],
        learning_starts=config["learning_starts"],
        batch_size=config["batch_size"],
        tau=config["tau"],
        gamma=config["gamma"],
        optimize_memory_usage=config["optimize_memory_usage"],
        replay_buffer_kwargs=dict(handle_timeout_termination=False),
        gradient_steps=config["gradient_steps"],
        train_freq=config["train_freq"],
        ent_coef=f"auto_{config['ent_coef_init']}",
        target_entropy=-float(train_env.action_space.shape[0]),
        seed=config["seed"],
        verbose=0,
        device=device,
        tensorboard_log=str(config["run_dir"] / "tb_logs"),
    )

    policy_kwargs = dict(
        net_arch=config["net_arch"],
        activation_fn=torch.nn.ReLU,
        optimizer_kwargs=dict(eps=1e-5),
        embed_dim=config["embed_dim"],
        num_heads=config["num_heads"],
        contact_hidden=config["contact_hidden"],
    )

    if experiment == "vanilla_sac":
        model = SAC(policy="MlpPolicy", policy_kwargs=dict(net_arch=config["net_arch"]), **common_kwargs)
    elif experiment == "attentive_sac":
        model = AttentiveSAC(policy=AttentiveSACPolicy, policy_kwargs=policy_kwargs, **common_kwargs)
    elif experiment == "kin_only_no_attn":
        model = SAC(
            policy=KinematicsOnlySACPolicy,
            policy_kwargs=dict(
                net_arch=config["net_arch"],
                activation_fn=torch.nn.ReLU,
                optimizer_kwargs=dict(eps=1e-5),
            ),
            **common_kwargs,
        )
    elif experiment == "full_obs_attn_loss":
        model = FullObsAttentionLossSAC(
            policy=FullObsAttentionLossPolicy,
            policy_kwargs=policy_kwargs,
            **common_kwargs,
        )
    elif experiment == "kin_root_attentive":
        model = AttentiveSAC(policy=KinRootAttentiveSACPolicy, policy_kwargs=policy_kwargs, **common_kwargs)
    elif experiment == "centered_attentive":
        model = AttentiveSAC(policy=CenteredAttentiveSACPolicy, policy_kwargs=policy_kwargs, **common_kwargs)
    elif experiment == "normalized_attentive":
        model = AttentiveSAC(
            policy=NormalizedAttentiveSACPolicy,
            policy_kwargs=policy_kwargs,
            **common_kwargs,
        )
    elif experiment == "kin_root_normalized_attentive":
        model = AttentiveSAC(
            policy=KinRootNormalizedAttentiveSACPolicy,
            policy_kwargs=policy_kwargs,
            **common_kwargs,
        )
    else:
        raise ValueError(f"Unsupported experiment '{experiment}'")

    if hasattr(model, "max_grad_norm"):
        model.max_grad_norm = config["max_grad_norm"]
    return model


def count_total_params(model) -> int:
    return sum(p.numel() for p in model.policy.parameters())


def save_summary(run_dir: Path, model, callback: MetricsCallback, config: Dict) -> Dict:
    summary = {
        "experiment": config["experiment"],
        "seed": config["seed"],
        "peak_reward": float(np.nanmax(callback.mean_rewards)) if callback.mean_rewards else None,
        "reward_at_500k": interpolate_metric_at_step(callback.timesteps, callback.mean_rewards, 500_000),
        "reward_at_1M": interpolate_metric_at_step(callback.timesteps, callback.mean_rewards, 1_000_000),
        "mean_fps": float(np.nanmean(callback.fps_history)) if callback.fps_history else None,
        "total_params": int(count_total_params(model)),
        "best_step": int(callback.best_step),
        "best_mean_reward": float(callback.best_mean_reward) if np.isfinite(callback.best_mean_reward) else None,
        "peak_ram_gb": float(np.nanmax(callback.ram_history)) if callback.ram_history else ram_gb(),
        "final_step": int(callback.timesteps[-1]) if callback.timesteps else 0,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def plot_curves(run_dir: Path, callback: MetricsCallback, config: Dict) -> None:
    if not callback.timesteps:
        return

    x = np.array(callback.timesteps) / 1000.0
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(f"{config['experiment']} | seed {config['seed']}", fontsize=14)

    axes[0, 0].plot(x, callback.mean_rewards, color="#0f766e")
    axes[0, 0].fill_between(
        x,
        np.array(callback.mean_rewards) - np.array(callback.std_rewards),
        np.array(callback.mean_rewards) + np.array(callback.std_rewards),
        color="#5eead4",
        alpha=0.3,
    )
    axes[0, 0].set_title("Reward")
    axes[0, 0].set_xlabel("Timesteps (k)")

    axes[0, 1].plot(x, callback.actor_losses, label="actor", color="#dc2626")
    axes[0, 1].plot(x, callback.critic_losses, label="critic", color="#2563eb")
    axes[0, 1].set_title("Losses")
    axes[0, 1].set_xlabel("Timesteps (k)")
    axes[0, 1].legend()

    axes[0, 2].plot(x, callback.q_values, color="#7c3aed")
    axes[0, 2].set_title("Mean Q")
    axes[0, 2].set_xlabel("Timesteps (k)")

    axes[1, 0].plot(x, callback.ent_coefs, color="#ea580c")
    axes[1, 0].set_title("Entropy Coef")
    axes[1, 0].set_xlabel("Timesteps (k)")

    axes[1, 1].plot(x, callback.fps_history, color="#1d4ed8")
    axes[1, 1].set_title("FPS")
    axes[1, 1].set_xlabel("Timesteps (k)")

    if any(np.isfinite(v) for v in callback.attn_means):
        axes[1, 2].plot(x, callback.attn_means, label="attn mean", color="#059669")
        axes[1, 2].plot(x, callback.attn_stds, label="attn std", color="#a16207")
        axes[1, 2].legend()
        axes[1, 2].set_title("Attention Stats")
    else:
        axes[1, 2].plot(x, callback.success_rates, color="#0891b2")
        axes[1, 2].set_title("Success Rate")
    axes[1, 2].set_xlabel("Timesteps (k)")

    for ax in axes.flat:
        ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(run_dir / "curves.png", bbox_inches="tight")
    plt.close(fig)


def plot_all_metrics(run_dir: Path, callback: MetricsCallback, config: Dict) -> None:
    if not callback.timesteps:
        return

    x = np.array(callback.timesteps) / 1000.0
    series = [
        ("Mean Reward", callback.mean_rewards, "#0f766e"),
        ("Std Reward", callback.std_rewards, "#14b8a6"),
        ("Min Reward", callback.min_rewards, "#ef4444"),
        ("Max Reward", callback.max_rewards, "#22c55e"),
        ("Episode Length", callback.ep_lengths, "#2563eb"),
        ("Actor Loss", callback.actor_losses, "#dc2626"),
        ("Critic Loss", callback.critic_losses, "#7c3aed"),
        ("Mean Q", callback.q_values, "#8b5cf6"),
        ("Entropy Coef", callback.ent_coefs, "#ea580c"),
        ("FPS", callback.fps_history, "#1d4ed8"),
        ("RAM (GB)", callback.ram_history, "#475569"),
        ("Success Rate", callback.success_rates, "#0891b2"),
        ("Attention Mean", callback.attn_means, "#059669"),
        ("Attention Std", callback.attn_stds, "#a16207"),
    ]

    fig, axes = plt.subplots(4, 4, figsize=(18, 14))
    fig.suptitle(f"All Metrics | {config['experiment']} | seed {config['seed']}", fontsize=15)

    for ax, (title, values, color) in zip(axes.flat, series):
        arr = np.array(values, dtype=float)
        if np.isfinite(arr).any():
            ax.plot(x, arr, color=color, linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("Timesteps (k)")
        ax.grid(True, alpha=0.2)

    for ax in axes.flat[len(series) :]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(run_dir / "all_metrics.png", bbox_inches="tight")
    plt.close(fig)


def update_comparison_table(results_root: Path) -> None:
    rows = []
    for summary_path in results_root.glob("*/summary.json"):
        try:
            rows.append(json.loads(summary_path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with (results_root / "comparison_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def default_run_name(config: Dict) -> str:
    experiment = config["experiment"]
    seed = config["seed"]
    if experiment == "attentive_sac":
        return f"attentive_sac_seed{seed}"
    if experiment == "vanilla_sac":
        return f"vanilla_sac_seed{seed}"
    if experiment == "kin_only_no_attn":
        return f"ablation_kin_only_no_attn_seed{seed}"
    if experiment == "full_obs_attn_loss":
        return f"ablation_full_obs_attn_loss_seed{seed}"
    if experiment == "kin_root_attentive":
        return f"refine_kin_root_attentive_seed{seed}"
    if experiment == "centered_attentive":
        return f"refine_centered_attentive_seed{seed}"
    if experiment == "normalized_attentive":
        return f"refine_normalized_attentive_seed{seed}"
    if experiment == "kin_root_normalized_attentive":
        return f"refine_kin_root_normalized_attentive_seed{seed}"
    return f"{experiment}_seed{seed}"


def run_experiment(config: Dict) -> Dict:
    results_root = ensure_dir(Path(config["results_root"]))
    run_dir = ensure_dir(results_root / config["run_name"])
    config["run_dir"] = run_dir

    (run_dir / "config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
    before_video_info = record_random_video(
        config,
        run_dir / "before_training.mp4",
        n_episodes=config["video_episodes"],
        max_steps_per_ep=config["video_max_steps"],
    )
    (run_dir / "before_video.json").write_text(json.dumps(before_video_info, indent=2), encoding="utf-8")

    train_env = build_vec_env(config)
    model = build_model(config["experiment"], train_env, config, config["device"])
    callback = MetricsCallback(
        run_dir=run_dir,
        log_interval=config["log_interval"],
        reward_threshold=config["reward_threshold"],
        checkpoint_freq=config["checkpoint_freq"],
        best_model_window=config["best_model_window"],
        alpha_min=config["alpha_min"],
        verbose=1,
    )

    start = time.time()
    try:
        model.learn(
            total_timesteps=config["total_timesteps"],
            callback=callback,
            log_interval=1,
            progress_bar=False,
        )
    except Exception as exc:
        crash_log = "\n".join(
            [
                f"experiment: {config['experiment']}",
                f"run_name: {config['run_name']}",
                f"timesteps_logged: {callback.timesteps[-1] if callback.timesteps else 0}",
                f"error_type: {type(exc).__name__}",
                f"error_message: {exc}",
            ]
        )
        (run_dir / "crash_log.txt").write_text(crash_log, encoding="utf-8")
        raise
    finally:
        elapsed = time.time() - start
        model.save(str(run_dir / "final_model"))
        train_env.save(str(run_dir / "final_vecnorm.pkl"))
        config["elapsed_seconds"] = elapsed
        (run_dir / "config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
        train_env.close()

    summary = save_summary(run_dir, model, callback, config)
    plot_curves(run_dir, callback, config)
    plot_all_metrics(run_dir, callback, config)

    vecnorm_path = run_dir / "best_model_vecnorm.pkl"
    model_path = run_dir / "best_model.zip"
    if model_path.exists() and vecnorm_path.exists():
        trained_video_info = record_trained_video(
            model,
            vecnorm_path,
            config,
            run_dir / "after_training.mp4",
            deterministic=True,
            n_episodes=config["video_episodes"],
            max_steps_per_ep=config["video_max_steps"],
        )
        (run_dir / "after_video.json").write_text(json.dumps(trained_video_info, indent=2), encoding="utf-8")

    update_comparison_table(results_root)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Humanoid-v4 research runner")
    parser.add_argument(
        "--experiment",
        choices=[
            "attentive_sac",
            "vanilla_sac",
            "kin_only_no_attn",
            "full_obs_attn_loss",
            "kin_root_attentive",
            "centered_attentive",
            "normalized_attentive",
            "kin_root_normalized_attentive",
        ],
        default="attentive_sac",
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--results-root", default=r"S:\rl_humanoid_runs\results")
    parser.add_argument("--env-id", default="Humanoid-v4")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-timesteps", type=int, default=100000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--buffer-size", type=int, default=500000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-starts", type=int, default=10000)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--train-freq", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--alpha-min", type=float, default=0.05)
    parser.add_argument("--ent-coef-init", type=float, default=0.1)
    parser.add_argument("--clip-obs", type=float, default=5.0)
    parser.add_argument("--clip-reward", type=float, default=10.0)
    parser.add_argument("--log-interval", type=int, default=5000)
    parser.add_argument("--checkpoint-freq", type=int, default=25000)
    parser.add_argument("--best-model-window", type=int, default=50)
    parser.add_argument("--reward-threshold", type=float, default=3000.0)
    parser.add_argument("--optimize-memory-usage", action="store_true")
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--contact-hidden", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--video-episodes", type=int, default=2)
    parser.add_argument("--video-max-steps", type=int, default=1000)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = vars(args)
    config["device"] = get_device(config["device"])
    config["net_arch"] = [512, 512]
    config["run_name"] = config["run_name"] or default_run_name(config)

    print(f"device={config['device']} torch={torch.__version__} cuda={torch.cuda.is_available()}")
    print(f"env={config['env_id']} experiment={config['experiment']} run={config['run_name']}")
    print(f"results_root={config['results_root']}")
    print(f"obs split root={len(ROOT_IDX)} kin={len(KIN_IDX)} contact={len(CONTACT_IDX)}")

    summary = run_experiment(config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
