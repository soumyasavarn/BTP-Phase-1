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
    list(range(5, 22)) + list(range(28, 45)) + list(range(269, 292)), dtype=torch.long)
CONTACT_IDX = torch.tensor(
    list(range(45, 185)) + list(range(185, 269)) + list(range(292, 376)), dtype=torch.long)
KIN_ROOT_IDX = torch.tensor(
    sorted(set(ROOT_IDX.tolist() + KIN_IDX.tolist())), dtype=torch.long)
LOG_STD_MAX, LOG_STD_MIN = 2, -20


def ram_gb():
    return psutil.virtual_memory().used / 1e9

def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_device(device_arg):
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg

def make_env(env_id, seed):
    def _init():
        env = gym.make(env_id)
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init

def build_vec_env(config):
    base = DummyVecEnv([make_env(config["env_id"], config["seed"])])
    return VecNormalize(base, norm_obs=True, norm_reward=True,
                        clip_obs=config["clip_obs"], clip_reward=config["clip_reward"],
                        gamma=config["gamma"])

def make_render_env(env_id, seed):
    env = gym.make(env_id, render_mode="rgb_array")
    env.reset(seed=seed)
    return env

def interpolate_metric_at_step(steps, values, target_step):
    if not steps: return None
    pairs = [(s, v) for s, v in zip(steps, values) if np.isfinite(v)]
    if not pairs: return None
    sa = np.array([p[0] for p in pairs], dtype=float)
    va = np.array([p[1] for p in pairs], dtype=float)
    if target_step <= sa[0]: return float(va[0])
    if target_step > sa[-1]: return None
    return float(np.interp(target_step, sa, va))


class KinematicsActor(SB3Actor):
    def __init__(self, observation_space, action_space, net_arch, features_extractor,
                 features_dim, activation_fn=nn.ReLU, **kwargs):
        super().__init__(observation_space, action_space, net_arch,
                         features_extractor, features_dim, activation_fn, **kwargs)
        kin_dim = len(KIN_IDX); act_dim = action_space.shape[0]
        layers, prev = [], kin_dim
        for h in net_arch:
            layers += [nn.Linear(prev, h), activation_fn()]
            prev = h
        self.latent_pi = nn.Sequential(*layers)
        self.mu = nn.Linear(prev, act_dim)
        self.log_std = nn.Linear(prev, act_dim)

    def get_action_dist_params(self, obs):
        kin = obs[:, KIN_IDX.to(obs.device)]
        latent = self.latent_pi(kin)
        mean = self.mu(latent)
        log_std = self.log_std(latent).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std, {}


class KinematicsRootActor(SB3Actor):
    def __init__(self, observation_space, action_space, net_arch, features_extractor,
                 features_dim, activation_fn=nn.ReLU, **kwargs):
        super().__init__(observation_space, action_space, net_arch,
                         features_extractor, features_dim, activation_fn, **kwargs)
        input_dim = len(KIN_ROOT_IDX); act_dim = action_space.shape[0]
        layers, prev = [], input_dim
        for h in net_arch:
            layers += [nn.Linear(prev, h), activation_fn()]
            prev = h
        self.latent_pi = nn.Sequential(*layers)
        self.mu = nn.Linear(prev, act_dim)
        self.log_std = nn.Linear(prev, act_dim)

    def get_action_dist_params(self, obs):
        kin_root = obs[:, KIN_ROOT_IDX.to(obs.device)]
        latent = self.latent_pi(kin_root)
        mean = self.mu(latent)
        log_std = self.log_std(latent).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std, {}


class FiLMActor(SB3Actor):
    '''Actor conditioned via FiLM: kin(57) through MLP, each layer
    modulated by root+contact cross-attention context.'''
    def __init__(self, observation_space, action_space, net_arch, features_extractor,
                 features_dim, activation_fn=nn.ReLU, film_dim=64, **kwargs):
        super().__init__(observation_space, action_space, net_arch,
                         features_extractor, features_dim, activation_fn, **kwargs)
        act_dim = action_space.shape[0]
        self._act_fn = activation_fn()
        self.context_encoder = nn.Sequential(nn.Linear(len(ROOT_IDX), film_dim), nn.ReLU())
        self.contact_enc = nn.Sequential(
            nn.Linear(len(CONTACT_IDX), 256), nn.ReLU(),
            nn.Linear(256, film_dim), nn.ReLU())
        self.cross_attn = nn.MultiheadAttention(film_dim, 4, batch_first=True, dropout=0.0)
        self.ctx_norm = nn.LayerNorm(film_dim)
        self.mlp_layers = nn.ModuleList()
        self.film_gamma = nn.ModuleList()
        self.film_beta  = nn.ModuleList()
        prev = len(KIN_IDX)
        for h in net_arch:
            self.mlp_layers.append(nn.Linear(prev, h))
            self.film_gamma.append(nn.Linear(film_dim, h))
            self.film_beta.append( nn.Linear(film_dim, h))
            prev = h
        self.mu = nn.Linear(prev, act_dim)
        self.log_std = nn.Linear(prev, act_dim)
        self.latent_pi = nn.Identity()
        print(f"  FiLMActor | kin({len(KIN_IDX)})+ctx({film_dim}) FiLM x{len(net_arch)} | params:{sum(p.numel() for p in self.parameters()):,}")

    def _context(self, obs):
        dev = obs.device
        r = obs[:, ROOT_IDX.to(dev)]; c = obs[:, CONTACT_IDX.to(dev)]
        r_e = self.context_encoder(r); c_e = self.contact_enc(c)
        att, _ = self.cross_attn(r_e.unsqueeze(1), c_e.unsqueeze(1), c_e.unsqueeze(1))
        return self.ctx_norm(att.squeeze(1) + r_e)

    def get_action_dist_params(self, obs):
        h = obs[:, KIN_IDX.to(obs.device)]
        ctx = self._context(obs)
        for lin, gfc, bfc in zip(self.mlp_layers, self.film_gamma, self.film_beta):
            h = self._act_fn(gfc(ctx) * lin(h) + bfc(ctx))
        return self.mu(h), self.log_std(h).clamp(LOG_STD_MIN, LOG_STD_MAX), {}


class RichContactAttention(nn.Module):
    """
    Multi-token contact attention producing three outputs:
      context     (B, embed_dim)  — injected into actor as extra input
      loss_weight (B, 1)          — normalized per-sample scalar for actor loss weighting
      entropy     (B, 1)          — attention entropy over contact tokens (diagnostic)

    Key difference from ContactAttentionWeighter:
    - Contact is tokenized into K separate tokens instead of 1, so attention can
      focus on specific contact 'aspects' rather than a global summary
    - The context vector is USED by the actor (not just the scalar weight)
    - Attention gets dual gradient signal: from how context shapes actions AND from loss weighting
    """
    def __init__(self, root_dim=11, contact_dim=308, embed_dim=64, num_heads=4,
                 contact_hidden=256, num_tokens=8, attn_scale=0.5):
        super().__init__()
        self.num_tokens = num_tokens
        self.embed_dim = embed_dim
        self.attn_scale = attn_scale
        self.contact_encoder = nn.Sequential(
            nn.Linear(contact_dim, contact_hidden), nn.ReLU(),
            nn.Linear(contact_hidden, num_tokens * embed_dim))
        self.root_proj = nn.Linear(root_dim, embed_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=0.0)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.weight_head = nn.Sequential(nn.Linear(embed_dim, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, obs):
        dev = obs.device
        root = obs[:, ROOT_IDX.to(dev)]; contact = obs[:, CONTACT_IDX.to(dev)]
        B = root.shape[0]
        c_tokens = self.contact_encoder(contact).view(B, self.num_tokens, self.embed_dim)
        r_emb = self.root_proj(root)
        attn_out, attn_weights = self.cross_attn(
            r_emb.unsqueeze(1), c_tokens, c_tokens, need_weights=True, average_attn_weights=True)
        context = self.layer_norm(attn_out.squeeze(1) + r_emb)
        raw = self.weight_head(context)
        c = raw - raw.mean(dim=0, keepdim=True)
        loss_weight = torch.exp(self.attn_scale * torch.tanh(c))
        loss_weight = loss_weight / loss_weight.mean(dim=0, keepdim=True).detach().clamp_min(1e-6)
        aw = attn_weights.squeeze(1).clamp(min=1e-8)
        entropy = -(aw * aw.log()).sum(dim=-1, keepdim=True)
        return context, loss_weight, entropy


class RichContactActor(SB3Actor):
    """
    Actor that conditions on learned contact context (not just loss-weighted by it).
    Input = kin(57) || attn_context(embed_dim) -> [512, 512] -> action.
    The RichContactAttention module lives here so context is available at inference time.
    """
    def __init__(self, observation_space, action_space, net_arch, features_extractor,
                 features_dim, activation_fn=nn.ReLU,
                 embed_dim=64, num_heads=4, contact_hidden=256, num_contact_tokens=8,
                 attn_scale=0.5, **kwargs):
        super().__init__(observation_space, action_space, net_arch,
                         features_extractor, features_dim, activation_fn, **kwargs)
        act_dim = action_space.shape[0]
        self.rich_attn = RichContactAttention(
            root_dim=len(ROOT_IDX), contact_dim=len(CONTACT_IDX),
            embed_dim=embed_dim, num_heads=num_heads,
            contact_hidden=contact_hidden, num_tokens=num_contact_tokens, attn_scale=attn_scale)
        actor_in = len(KIN_IDX) + embed_dim
        layers, prev = [], actor_in
        for h in net_arch:
            layers += [nn.Linear(prev, h), activation_fn()]
            prev = h
        self.latent_pi = nn.Sequential(*layers)
        self.mu = nn.Linear(prev, act_dim)
        self.log_std = nn.Linear(prev, act_dim)
        print(f"  RichContactActor | in:{actor_in}(kin:{len(KIN_IDX)}+ctx:{embed_dim}) "
              f"K={num_contact_tokens} tokens | params:{sum(p.numel() for p in self.parameters()):,}")

    def get_action_dist_params(self, obs):
        kin = obs[:, KIN_IDX.to(obs.device)]
        context, _, _ = self.rich_attn(obs)
        h = torch.cat([kin, context], dim=-1)
        latent = self.latent_pi(h)
        return self.mu(latent), self.log_std(latent).clamp(LOG_STD_MIN, LOG_STD_MAX), {}


class ContactAttentionWeighter(nn.Module):
    def __init__(self, root_dim=11, contact_dim=308, embed_dim=64, num_heads=4,
                 contact_hidden=256, weight_mode="sigmoid2x", attn_scale=0.25):
        super().__init__()
        self.weight_mode = weight_mode; self.attn_scale = attn_scale
        self.contact_encoder = nn.Sequential(
            nn.Linear(contact_dim, contact_hidden), nn.ReLU(),
            nn.Linear(contact_hidden, embed_dim), nn.ReLU())
        self.root_proj = nn.Sequential(nn.Linear(root_dim, embed_dim), nn.ReLU())
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=0.0)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.weight_head = nn.Sequential(nn.Linear(embed_dim, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, obs):
        dev = obs.device
        root = obs[:, ROOT_IDX.to(dev)]; contact = obs[:, CONTACT_IDX.to(dev)]
        r_emb = self.root_proj(root); c_emb = self.contact_encoder(contact)
        att, _ = self.cross_attn(r_emb.unsqueeze(1), c_emb.unsqueeze(1), c_emb.unsqueeze(1))
        att = self.layer_norm(att.squeeze(1) + r_emb)
        raw = self.weight_head(att)
        if self.weight_mode == "centered_tanh":
            c = raw - raw.mean(dim=0, keepdim=True)
            return 1.0 + self.attn_scale * torch.tanh(c)
        if self.weight_mode == "normalized_exp":
            c = raw - raw.mean(dim=0, keepdim=True)
            pos = torch.exp(self.attn_scale * torch.tanh(c))
            return pos / pos.mean(dim=0, keepdim=True).detach().clamp_min(1e-6)
        return 2.0 * torch.sigmoid(raw)


class KinematicsOnlySACPolicy(SACPolicy):
    def make_actor(self, fe=None):
        return KinematicsActor(**self._update_features_extractor(self.actor_kwargs, fe)).to(self.device)

class AttentiveSACPolicy(SACPolicy):
    def __init__(self, *a, embed_dim=64, num_heads=4, contact_hidden=256,
                 weight_mode="sigmoid2x", attn_scale=0.25, **kw):
        self._embed_dim=embed_dim; self._num_heads=num_heads
        self._contact_hidden=contact_hidden; self._weight_mode=weight_mode; self._attn_scale=attn_scale
        super().__init__(*a, **kw)
    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        self.attn_weighter = ContactAttentionWeighter(
            embed_dim=self._embed_dim, num_heads=self._num_heads,
            contact_hidden=self._contact_hidden,
            weight_mode=self._weight_mode, attn_scale=self._attn_scale).to(self.device)
        self.actor.optimizer = self.optimizer_class(
            list(self.actor.parameters()) + list(self.attn_weighter.parameters()),
            lr=lr_schedule(1), **self.optimizer_kwargs)
    def make_actor(self, fe=None):
        return KinematicsActor(**self._update_features_extractor(self.actor_kwargs, fe)).to(self.device)

class FullObsAttentionLossPolicy(AttentiveSACPolicy):
    def make_actor(self, fe=None):
        return super(SACPolicy, self).make_actor(fe)  # default SACPolicy actor (full obs)

class KinRootAttentiveSACPolicy(AttentiveSACPolicy):
    def make_actor(self, fe=None):
        return KinematicsRootActor(**self._update_features_extractor(self.actor_kwargs, fe)).to(self.device)

class CenteredAttentiveSACPolicy(AttentiveSACPolicy):
    def __init__(self, *a, **kw):
        kw.setdefault("weight_mode","centered_tanh"); kw.setdefault("attn_scale",0.25)
        super().__init__(*a, **kw)

class NormalizedAttentiveSACPolicy(AttentiveSACPolicy):
    def __init__(self, *a, **kw):
        kw.setdefault("weight_mode","normalized_exp"); kw.setdefault("attn_scale",0.50)
        super().__init__(*a, **kw)

class KinRootNormalizedAttentiveSACPolicy(NormalizedAttentiveSACPolicy):
    def make_actor(self, fe=None):
        return KinematicsRootActor(**self._update_features_extractor(self.actor_kwargs, fe)).to(self.device)

class FiLMSACPolicy(SACPolicy):
    def __init__(self, *a, film_dim=64, **kw):
        self._film_dim = film_dim; super().__init__(*a, **kw)
    def make_actor(self, fe=None):
        return FiLMActor(**self._update_features_extractor(self.actor_kwargs, fe),
                         film_dim=self._film_dim).to(self.device)

class RichContactAttentiveSACPolicy(SACPolicy):
    def __init__(self, *a, embed_dim=64, num_heads=4, contact_hidden=256,
                 num_contact_tokens=8, attn_scale=0.5, **kw):
        self._rc_embed_dim = embed_dim; self._rc_num_heads = num_heads
        self._rc_contact_hidden = contact_hidden; self._rc_num_tokens = num_contact_tokens
        self._rc_attn_scale = attn_scale
        super().__init__(*a, **kw)
    def make_actor(self, fe=None):
        kw = self._update_features_extractor(self.actor_kwargs, fe)
        kw.update(embed_dim=self._rc_embed_dim, num_heads=self._rc_num_heads,
                  contact_hidden=self._rc_contact_hidden, num_contact_tokens=self._rc_num_tokens,
                  attn_scale=self._rc_attn_scale)
        return RichContactActor(**kw).to(self.device)


class AttentionLossMixin:
    '''
    Overrides SAC.train() with attention-weighted actor loss.
    Key fix: attn_grad_clip (default 1.0) clips attention gradients
    SEPARATELY from the actor (max_grad_norm=10.0) to prevent collapse.
    '''
    def _get_log_ent_coef(self):
        for attr in ("log_ent_coef", "log_ent_coef_param"):
            if hasattr(self, attr): return getattr(self, attr)
        params = self.ent_coef_optimizer.param_groups[0].get("params", [])
        return params[0] if params else None

    def train(self, gradient_steps, batch_size=64):
        self.policy.set_training_mode(True)
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer, self.ent_coef_optimizer])
        grad_clip      = getattr(self, "max_grad_norm",  10.0)
        attn_grad_clip = getattr(self, "attn_grad_clip",  1.0)

        al, cl, ecl, ecs, qs = [], [], [], [], []
        am, astd, amin, amax = [], [], [], []
        log_ec = self._get_log_ent_coef()

        for _ in range(gradient_steps):
            self._n_updates += 1
            rd  = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            obs = rd.observations

            with torch.no_grad():
                _, lp = self.actor.action_log_prob(obs)
                lp = lp.reshape(-1,1)
            ec = torch.exp(log_ec.detach())
            ec_loss = -(log_ec * (lp + self.target_entropy).detach()).mean()
            self.ent_coef_optimizer.zero_grad(); ec_loss.backward(); self.ent_coef_optimizer.step()
            ecl.append(ec_loss.item()); ecs.append(ec.item())

            with torch.no_grad():
                na, nlp = self.actor.action_log_prob(rd.next_observations)
                nq = torch.cat(self.critic_target(rd.next_observations, na), dim=1)
                nqm, _ = torch.min(nq, dim=1, keepdim=True)
                tq = rd.rewards + (1-rd.dones)*self.gamma*(nqm - ec*nlp.reshape(-1,1))
            cq = self.critic(obs, rd.actions)
            c_loss = sum(F.mse_loss(q, tq) for q in cq)
            self.critic.optimizer.zero_grad(); c_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), grad_clip)
            self.critic.optimizer.step(); cl.append(c_loss.item())

            for p in self.critic.parameters(): p.requires_grad_(False)
            api, lp = self.actor.action_log_prob(obs); lp = lp.reshape(-1,1)
            qpi = torch.cat(self.critic(obs, api), dim=1)
            mq, _ = torch.min(qpi, dim=1, keepdim=True)
            w = self.policy.attn_weighter(obs)
            a_loss = (w * (ec*lp - mq)).mean()
            self.actor.optimizer.zero_grad(); a_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), grad_clip)
            torch.nn.utils.clip_grad_norm_(self.policy.attn_weighter.parameters(), attn_grad_clip)
            self.actor.optimizer.step()
            al.append(a_loss.item()); qs.append(mq.mean().item())
            am.append(w.mean().item()); astd.append(w.std(unbiased=False).item())
            amin.append(w.min().item()); amax.append(w.max().item())
            for p in self.critic.parameters(): p.requires_grad_(True)
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.mul_(1-self.tau).add_(self.tau*p.data)

        self.logger.record("train/n_updates",     self._n_updates)
        self.logger.record("train/actor_loss",    np.mean(al))
        self.logger.record("train/critic_loss",   np.mean(cl))
        self.logger.record("train/ent_coef",      np.mean(ecs))
        self.logger.record("train/ent_coef_loss", np.mean(ecl))
        self.logger.record("train/mean_q1",       np.mean(qs))
        self.logger.record("train/attn_mean",     np.mean(am))
        self.logger.record("train/attn_std",      np.mean(astd))
        self.logger.record("train/attn_min",      np.mean(amin))
        self.logger.record("train/attn_max",      np.mean(amax))


class AttentiveSAC(AttentionLossMixin, SAC): pass
class FullObsAttentionLossSAC(AttentionLossMixin, SAC): pass
class FiLMSAC(SAC): pass   # standard loss; FiLM lives inside the actor


class RichAttentionLossMixin(AttentionLossMixin):
    """
    SAC.train() override for RichContactActor.
    The attention module is INSIDE the actor (not on the policy), so:
    - actor.action_log_prob(obs) uses the context internally (gradient path 1)
    - actor.rich_attn(obs) called separately to get loss_weight (gradient path 2)
    Attention therefore gets gradient signal from BOTH how it conditions behavior
    AND how it weights the loss — the key advantage over AttentionLossMixin.
    """
    def train(self, gradient_steps, batch_size=64):
        self.policy.set_training_mode(True)
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer, self.ent_coef_optimizer])
        grad_clip      = getattr(self, "max_grad_norm", 10.0)
        attn_grad_clip = getattr(self, "attn_grad_clip",  1.0)

        al, cl, ecl, ecs, qs = [], [], [], [], []
        am, astd, amin, amax, aent = [], [], [], [], []
        log_ec = self._get_log_ent_coef()

        for _ in range(gradient_steps):
            self._n_updates += 1
            rd  = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            obs = rd.observations

            with torch.no_grad():
                _, lp = self.actor.action_log_prob(obs)
                lp = lp.reshape(-1, 1)
            ec = torch.exp(log_ec.detach())
            ec_loss = -(log_ec * (lp + self.target_entropy).detach()).mean()
            self.ent_coef_optimizer.zero_grad(); ec_loss.backward(); self.ent_coef_optimizer.step()
            ecl.append(ec_loss.item()); ecs.append(ec.item())

            with torch.no_grad():
                na, nlp = self.actor.action_log_prob(rd.next_observations)
                nq = torch.cat(self.critic_target(rd.next_observations, na), dim=1)
                nqm, _ = torch.min(nq, dim=1, keepdim=True)
                tq = rd.rewards + (1-rd.dones)*self.gamma*(nqm - ec*nlp.reshape(-1,1))
            cq = self.critic(obs, rd.actions)
            c_loss = sum(F.mse_loss(q, tq) for q in cq)
            self.critic.optimizer.zero_grad(); c_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), grad_clip)
            self.critic.optimizer.step(); cl.append(c_loss.item())

            for p in self.critic.parameters(): p.requires_grad_(False)
            # Path 1: action_log_prob internally calls rich_attn for context -> gradients to attn + MLP
            api, lp = self.actor.action_log_prob(obs); lp = lp.reshape(-1, 1)
            # Path 2: explicit rich_attn call for loss weighting -> additional gradients to attn
            _, loss_weight, entropy = self.actor.rich_attn(obs)
            qpi = torch.cat(self.critic(obs, api), dim=1)
            mq, _ = torch.min(qpi, dim=1, keepdim=True)
            a_loss = (loss_weight * (ec*lp - mq)).mean()
            self.actor.optimizer.zero_grad(); a_loss.backward()
            # Clip actor MLP and attention separately to prevent attn collapse
            actor_core = [p for n, p in self.actor.named_parameters() if not n.startswith('rich_attn')]
            torch.nn.utils.clip_grad_norm_(actor_core, grad_clip)
            torch.nn.utils.clip_grad_norm_(self.actor.rich_attn.parameters(), attn_grad_clip)
            self.actor.optimizer.step()
            al.append(a_loss.item()); qs.append(mq.mean().item())
            am.append(loss_weight.mean().item()); astd.append(loss_weight.std(unbiased=False).item())
            amin.append(loss_weight.min().item()); amax.append(loss_weight.max().item())
            aent.append(entropy.mean().item())
            for p in self.critic.parameters(): p.requires_grad_(True)
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.mul_(1-self.tau).add_(self.tau*p.data)

        self.logger.record("train/n_updates",     self._n_updates)
        self.logger.record("train/actor_loss",    np.mean(al))
        self.logger.record("train/critic_loss",   np.mean(cl))
        self.logger.record("train/ent_coef",      np.mean(ecs))
        self.logger.record("train/ent_coef_loss", np.mean(ecl))
        self.logger.record("train/mean_q1",       np.mean(qs))
        self.logger.record("train/attn_mean",     np.mean(am))
        self.logger.record("train/attn_std",      np.mean(astd))
        self.logger.record("train/attn_min",      np.mean(amin))
        self.logger.record("train/attn_max",      np.mean(amax))
        self.logger.record("train/attn_entropy",  np.mean(aent))


class RichContactAttentiveSAC(RichAttentionLossMixin, SAC): pass


# ═══════════════════════════════════════════════════════════════════════════════
# Method A: contact_additive_loss
#
# Contact forces kept SEPARATE from the actor network (not mixed into actor MLP).
# Attention over K contact tokens produces a 64-dim context vector.
# This vector is projected to action space and the cosine alignment is ADDED to
# the SAC loss — additive, not multiplicative. Full 64-dim contact info preserved.
# ═══════════════════════════════════════════════════════════════════════════════

class ContactCrossAttentionModule(nn.Module):
    """Pure contact-attention module, fully separated from the actor MLP.
    Produces a context vector (embed_dim) and attention entropy per sample."""
    def __init__(self, root_dim=11, contact_dim=308, embed_dim=64, num_heads=4,
                 contact_hidden=256, num_tokens=8):
        super().__init__()
        self.num_tokens = num_tokens; self.embed_dim = embed_dim
        self.contact_enc = nn.Sequential(
            nn.Linear(contact_dim, contact_hidden), nn.ReLU(),
            nn.Linear(contact_hidden, num_tokens * embed_dim))
        self.root_proj = nn.Linear(root_dim, embed_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=0.0)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, obs):
        dev = obs.device
        root = obs[:, ROOT_IDX.to(dev)]; contact = obs[:, CONTACT_IDX.to(dev)]
        B = root.shape[0]
        c_tokens = self.contact_enc(contact).view(B, self.num_tokens, self.embed_dim)
        r_emb = self.root_proj(root)
        attn_out, attn_weights = self.cross_attn(
            r_emb.unsqueeze(1), c_tokens, c_tokens, need_weights=True, average_attn_weights=True)
        context = self.layer_norm(attn_out.squeeze(1) + r_emb)
        aw = attn_weights.squeeze(1).clamp(min=1e-8)
        entropy = -(aw * aw.log()).sum(dim=-1, keepdim=True)
        return context, entropy


class ContactAdditiveSACPolicy(SACPolicy):
    """Contact forces handled by a SEPARATE ContactCrossAttentionModule.
    Actor uses KIN only. Contact enters the loss additively via cosine alignment."""
    def __init__(self, *a, embed_dim=64, num_heads=4, contact_hidden=256, num_tokens=8, **kw):
        self._ca_embed_dim = embed_dim; self._ca_num_heads = num_heads
        self._ca_contact_hidden = contact_hidden; self._ca_num_tokens = num_tokens
        super().__init__(*a, **kw)

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        act_dim = self.action_space.shape[0]
        self.contact_module = ContactCrossAttentionModule(
            embed_dim=self._ca_embed_dim, num_heads=self._ca_num_heads,
            contact_hidden=self._ca_contact_hidden, num_tokens=self._ca_num_tokens).to(self.device)
        self.contact_action_proj = nn.Linear(self._ca_embed_dim, act_dim).to(self.device)
        self.actor.optimizer = self.optimizer_class(
            list(self.actor.parameters()) +
            list(self.contact_module.parameters()) +
            list(self.contact_action_proj.parameters()),
            lr=lr_schedule(1), **self.optimizer_kwargs)

    def make_actor(self, fe=None):
        return KinematicsActor(**self._update_features_extractor(self.actor_kwargs, fe)).to(self.device)


class ContactAdditiveLossMixin(AttentionLossMixin):
    """
    actor_loss = (α*lp - Q).mean()                              ← SAC base
               + λ * (-cosine_sim(policy_action, contact_prior)) ← additive contact term

    contact_prior = contact_action_proj(contact_cross_attention(obs))  — full 64-dim used.
    """
    def train(self, gradient_steps, batch_size=64):
        self.policy.set_training_mode(True)
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer, self.ent_coef_optimizer])
        grad_clip      = getattr(self, "max_grad_norm", 10.0)
        attn_grad_clip = getattr(self, "attn_grad_clip", 1.0)
        align_lambda   = getattr(self, "contact_align_lambda", 0.1)

        al, cl, ecl, ecs, qs, aent, cal = [], [], [], [], [], [], []
        log_ec = self._get_log_ent_coef()

        for _ in range(gradient_steps):
            self._n_updates += 1
            rd = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            obs = rd.observations

            with torch.no_grad():
                _, lp = self.actor.action_log_prob(obs); lp = lp.reshape(-1, 1)
            ec = torch.exp(log_ec.detach())
            ec_loss = -(log_ec * (lp + self.target_entropy).detach()).mean()
            self.ent_coef_optimizer.zero_grad(); ec_loss.backward(); self.ent_coef_optimizer.step()
            ecl.append(ec_loss.item()); ecs.append(ec.item())

            with torch.no_grad():
                na, nlp = self.actor.action_log_prob(rd.next_observations)
                nq = torch.cat(self.critic_target(rd.next_observations, na), dim=1)
                nqm, _ = torch.min(nq, dim=1, keepdim=True)
                tq = rd.rewards + (1-rd.dones)*self.gamma*(nqm - ec*nlp.reshape(-1,1))
            cq = self.critic(obs, rd.actions)
            c_loss = sum(F.mse_loss(q, tq) for q in cq)
            self.critic.optimizer.zero_grad(); c_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), grad_clip)
            self.critic.optimizer.step(); cl.append(c_loss.item())

            for p in self.critic.parameters(): p.requires_grad_(False)
            api, lp = self.actor.action_log_prob(obs); lp = lp.reshape(-1, 1)
            qpi = torch.cat(self.critic(obs, api), dim=1)
            mq, _ = torch.min(qpi, dim=1, keepdim=True)
            # SAC base loss
            base_loss = (ec*lp - mq).mean()
            # Contact additive term: full 64-dim context → action-space cosine alignment
            ctx, entropy = self.policy.contact_module(obs)
            contact_prior = self.policy.contact_action_proj(ctx)   # (B, A)
            align_loss = -F.cosine_similarity(api, contact_prior, dim=-1).mean()
            a_loss = base_loss + align_lambda * align_loss
            self.actor.optimizer.zero_grad(); a_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), grad_clip)
            contact_params = list(self.policy.contact_module.parameters()) + \
                             list(self.policy.contact_action_proj.parameters())
            torch.nn.utils.clip_grad_norm_(contact_params, attn_grad_clip)
            self.actor.optimizer.step()
            al.append(a_loss.item()); qs.append(mq.mean().item())
            aent.append(entropy.mean().item()); cal.append(align_loss.item())
            for p in self.critic.parameters(): p.requires_grad_(True)
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.mul_(1-self.tau).add_(self.tau*p.data)

        self.logger.record("train/n_updates",          self._n_updates)
        self.logger.record("train/actor_loss",         np.mean(al))
        self.logger.record("train/critic_loss",        np.mean(cl))
        self.logger.record("train/ent_coef",           np.mean(ecs))
        self.logger.record("train/ent_coef_loss",      np.mean(ecl))
        self.logger.record("train/mean_q1",            np.mean(qs))
        self.logger.record("train/attn_mean",          np.nan)
        self.logger.record("train/attn_std",           np.nan)
        self.logger.record("train/attn_entropy",       np.mean(aent))
        self.logger.record("train/contact_align_loss", np.mean(cal))


class ContactAdditiveSAC(ContactAdditiveLossMixin, SAC): pass


# ═══════════════════════════════════════════════════════════════════════════════
# Method B: contact_per_token_loss
#
# K=8 INDEPENDENT sigmoid-activated contact channels (NOT softmax, NOT sum-to-1).
# Total loss weight per sample ∈ [0, K] — genuinely multi-dimensional.
# Each contact token independently amplifies/suppresses its sample in the loss,
# with K separate gradient paths back through the contact encoding.
# ═══════════════════════════════════════════════════════════════════════════════

class ContactTokenWeighter(nn.Module):
    """K independent sigmoid activations. Unlike softmax, no normalization — each
    channel freely activates based on its contact group, giving per-sample total
    weight in [0, K] instead of a fixed α=1.0 scalar."""
    def __init__(self, contact_dim=308, root_dim=11, num_tokens=8, contact_hidden=128):
        super().__init__()
        self.num_tokens = num_tokens
        self.contact_enc = nn.Sequential(
            nn.Linear(contact_dim, contact_hidden), nn.ReLU(),
            nn.Linear(contact_hidden, num_tokens))
        self.root_gate = nn.Sequential(
            nn.Linear(root_dim, 32), nn.ReLU(),
            nn.Linear(32, num_tokens))

    def forward(self, obs):
        dev = obs.device
        contact = obs[:, CONTACT_IDX.to(dev)]; root = obs[:, ROOT_IDX.to(dev)]
        c_scores = self.contact_enc(contact)           # (B, K) — unnormalized per channel
        r_gate   = torch.sigmoid(self.root_gate(root)) # (B, K) — [0,1] root gating
        return torch.sigmoid(c_scores) * r_gate        # (B, K) — NOT summing to 1


class ContactTokenSACPolicy(SACPolicy):
    def __init__(self, *a, num_contact_tokens=8, contact_token_hidden=128, **kw):
        self._ct_num_tokens = num_contact_tokens; self._ct_hidden = contact_token_hidden
        super().__init__(*a, **kw)

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        self.token_weighter = ContactTokenWeighter(
            num_tokens=self._ct_num_tokens, contact_hidden=self._ct_hidden).to(self.device)
        self.actor.optimizer = self.optimizer_class(
            list(self.actor.parameters()) + list(self.token_weighter.parameters()),
            lr=lr_schedule(1), **self.optimizer_kwargs)

    def make_actor(self, fe=None):
        return KinematicsActor(**self._update_features_extractor(self.actor_kwargs, fe)).to(self.device)


class ContactTokenLossMixin(AttentionLossMixin):
    """
    actor_loss = (token_weights(B,K) * advantage(B,1)).sum(K).mean(B)

    token_weights are K independent sigmoids (NOT normalized to sum 1).
    Per-sample total weight = Σ_k sigmoid_k ∈ [0, K].
    K separate gradient paths through the contact encoding — richer than single α.
    """
    def train(self, gradient_steps, batch_size=64):
        self.policy.set_training_mode(True)
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer, self.ent_coef_optimizer])
        grad_clip      = getattr(self, "max_grad_norm", 10.0)
        attn_grad_clip = getattr(self, "attn_grad_clip", 1.0)

        al, cl, ecl, ecs, qs, twm, twstd = [], [], [], [], [], [], []
        log_ec = self._get_log_ent_coef()

        for _ in range(gradient_steps):
            self._n_updates += 1
            rd = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            obs = rd.observations

            with torch.no_grad():
                _, lp = self.actor.action_log_prob(obs); lp = lp.reshape(-1, 1)
            ec = torch.exp(log_ec.detach())
            ec_loss = -(log_ec * (lp + self.target_entropy).detach()).mean()
            self.ent_coef_optimizer.zero_grad(); ec_loss.backward(); self.ent_coef_optimizer.step()
            ecl.append(ec_loss.item()); ecs.append(ec.item())

            with torch.no_grad():
                na, nlp = self.actor.action_log_prob(rd.next_observations)
                nq = torch.cat(self.critic_target(rd.next_observations, na), dim=1)
                nqm, _ = torch.min(nq, dim=1, keepdim=True)
                tq = rd.rewards + (1-rd.dones)*self.gamma*(nqm - ec*nlp.reshape(-1,1))
            cq = self.critic(obs, rd.actions)
            c_loss = sum(F.mse_loss(q, tq) for q in cq)
            self.critic.optimizer.zero_grad(); c_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), grad_clip)
            self.critic.optimizer.step(); cl.append(c_loss.item())

            for p in self.critic.parameters(): p.requires_grad_(False)
            api, lp = self.actor.action_log_prob(obs); lp = lp.reshape(-1, 1)
            qpi = torch.cat(self.critic(obs, api), dim=1)
            mq, _ = torch.min(qpi, dim=1, keepdim=True)
            # K-channel contact loss: (B,K) * (B,1) → sum K → mean B
            token_w = self.policy.token_weighter(obs)  # (B, K) — NOT softmax
            adv     = (ec*lp - mq)                     # (B, 1)
            a_loss  = (token_w * adv).sum(dim=1).mean()
            self.actor.optimizer.zero_grad(); a_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), grad_clip)
            torch.nn.utils.clip_grad_norm_(self.policy.token_weighter.parameters(), attn_grad_clip)
            self.actor.optimizer.step()
            al.append(a_loss.item()); qs.append(mq.mean().item())
            twm.append(token_w.mean().item()); twstd.append(token_w.std(unbiased=False).item())
            for p in self.critic.parameters(): p.requires_grad_(True)
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.mul_(1-self.tau).add_(self.tau*p.data)

        self.logger.record("train/n_updates",     self._n_updates)
        self.logger.record("train/actor_loss",    np.mean(al))
        self.logger.record("train/critic_loss",   np.mean(cl))
        self.logger.record("train/ent_coef",      np.mean(ecs))
        self.logger.record("train/ent_coef_loss", np.mean(ecl))
        self.logger.record("train/mean_q1",       np.mean(qs))
        self.logger.record("train/attn_mean",     np.mean(twm))
        self.logger.record("train/attn_std",      np.mean(twstd))
        self.logger.record("train/attn_entropy",  np.nan)
        self.logger.record("train/contact_align_loss", np.nan)


class ContactTokenSAC(ContactTokenLossMixin, SAC): pass


# ═══════════════════════════════════════════════════════════════════════════════
# Method 4: contact_joint_weight_loss  (NEW — professor's direction)
#
# Addresses the core problem: 308 contact dims → 1 scalar collapses too much.
#
# Instead: contact(308) → cross-attention with 17 JOINT QUERIES → 17 weights
# (one weight per action dimension).  Weights are softmax-normalised across joints
# so Σ_j w_j = 1 — no amplification, α stays calibrated.
#
# The per-joint weights modulate the ENTROPY component of the actor loss
# per action dimension, leaving Q-value gradients and α-update untouched.
#
# actor_loss = E[ Σ_j w_j(contact) * α * lp_j  -  min_Q ]
#               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#               17-dimensional contact structure in entropy term
#
# Stability fix vs. Methods A & B:
#   • Entropy coef update uses STANDARD scalar lp (not weighted) → α is well-calibrated
#   • Softmax Σ_j w_j = 1 → no gradient amplification
#   • Q-value term unweighted → critic gradients stable
# ═══════════════════════════════════════════════════════════════════════════════

class JointContactAttention(nn.Module):
    """
    17 learnable joint query embeddings attend to K=16 contact tokens.
    Produces a (B, 17) weight vector (softmax across joints per sample).

    Attention map shape (B, num_heads, 17, K) is the richer signal:
    each of 17 joints has an independent distribution over K contact groups.
    """
    def __init__(self, contact_dim=308, action_dim=17, embed_dim=64,
                 num_heads=4, num_contact_tokens=16):
        super().__init__()
        self.num_tokens = num_contact_tokens
        self.embed_dim  = embed_dim
        self.action_dim = action_dim
        self.joint_queries = nn.Embedding(action_dim, embed_dim)
        self.contact_enc   = nn.Sequential(
            nn.Linear(contact_dim, 256), nn.ReLU(),
            nn.Linear(256, num_contact_tokens * embed_dim))
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True, dropout=0.0)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.joint_scalar = nn.Linear(embed_dim, 1)

    def forward(self, obs):
        dev = obs.device
        contact = obs[:, CONTACT_IDX.to(dev)]
        B = contact.shape[0]
        c_tokens = self.contact_enc(contact).view(B, self.num_tokens, self.embed_dim)
        joint_q  = self.joint_queries.weight.unsqueeze(0).expand(B, -1, -1)
        attn_out, attn_weights = self.cross_attn(
            joint_q, c_tokens, c_tokens,
            need_weights=True, average_attn_weights=False)
        # attn_out:     (B, 17, embed_dim)
        # attn_weights: (B, num_heads, 17, K)
        attn_out = self.layer_norm(attn_out)
        per_joint_logits = self.joint_scalar(attn_out).squeeze(-1)  # (B, 17)
        joint_weights    = torch.softmax(per_joint_logits, dim=-1)  # (B, 17) sums to 1
        # Per-joint entropy over contact tokens (averaged across heads)
        aw = attn_weights.mean(dim=1).clamp(min=1e-8)   # (B, 17, K)
        per_joint_ent = -(aw * aw.log()).sum(dim=-1)     # (B, 17)
        mean_entropy  = per_joint_ent.mean()
        return joint_weights, mean_entropy               # (B,17), scalar


class ContactJointActor(SB3Actor):
    """
    KinematicsActor extended with action_log_prob_per_dim:
    returns (action, per_dim_log_prob) where per_dim_log_prob is (B, A)
    NOT summed across action dimensions.  Contact forces are never seen here.
    """
    def __init__(self, observation_space, action_space, net_arch, features_extractor,
                 features_dim, activation_fn=nn.ReLU, **kwargs):
        super().__init__(observation_space, action_space, net_arch,
                         features_extractor, features_dim, activation_fn, **kwargs)
        act_dim = action_space.shape[0]
        kin_dim = len(KIN_IDX)
        layers, prev = [], kin_dim
        for h in net_arch:
            layers += [nn.Linear(prev, h), activation_fn()]
            prev = h
        self.latent_pi = nn.Sequential(*layers)
        self.mu      = nn.Linear(prev, act_dim)
        self.log_std = nn.Linear(prev, act_dim)
        nn.init.constant_(self.log_std.bias, -1.0)

    def get_action_dist_params(self, obs):
        kin = obs[:, KIN_IDX.to(obs.device)]
        latent = self.latent_pi(kin)
        mean   = self.mu(latent)
        log_std = self.log_std(latent).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std, {}

    def action_log_prob_per_dim(self, obs):
        """Reparameterised sample + per-dimension log_prob (B, A) before sum."""
        mean, log_std, _ = self.get_action_dist_params(obs)
        std = log_std.exp()
        eps = torch.randn_like(mean)
        x_t    = mean + std * eps          # unsquashed
        action = torch.tanh(x_t)           # squashed (B, A)
        # log N(x_t; mean, std) per dim
        lp_gaussian = -0.5 * eps.pow(2) - log_std - 0.9189  # log(√(2π))≈0.9189
        # tanh correction per dim
        lp_correction = torch.log(1.0 - action.pow(2) + 1e-6)
        per_dim_lp = lp_gaussian - lp_correction             # (B, A)
        return action, per_dim_lp


class ContactJointWeightSACPolicy(SACPolicy):
    """Contact forces processed by JointContactAttention (separate from actor).
    Actor (ContactJointActor) uses KIN only."""
    def __init__(self, *a, embed_dim=64, num_heads=4, num_contact_tokens=16, **kw):
        self._cj_embed_dim = embed_dim
        self._cj_num_heads = num_heads
        self._cj_num_tokens = num_contact_tokens
        super().__init__(*a, **kw)

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        act_dim = self.action_space.shape[0]
        self.joint_attn = JointContactAttention(
            action_dim=act_dim,
            embed_dim=self._cj_embed_dim,
            num_heads=self._cj_num_heads,
            num_contact_tokens=self._cj_num_tokens).to(self.device)
        self.actor.optimizer = self.optimizer_class(
            list(self.actor.parameters()) + list(self.joint_attn.parameters()),
            lr=lr_schedule(1), **self.optimizer_kwargs)

    def make_actor(self, fe=None):
        return ContactJointActor(
            **self._update_features_extractor(self.actor_kwargs, fe)).to(self.device)


class ContactJointLossMixin(AttentionLossMixin):
    """
    Entropy coef update: standard scalar lp  → α stays well-calibrated.
    Actor update:        per-dim weighted lp  → 17-dim contact structure.

    actor_loss = E[ Σ_{j=1}^{17} w_j(contact) · α · lp_j(s,a) − min_Q(s,a) ]

    w_j softmax-normalised  ⟹  Σ_j w_j = 1  ⟹  no gradient amplification.
    """
    def train(self, gradient_steps, batch_size=64):
        self.policy.set_training_mode(True)
        self._update_learning_rate(
            [self.actor.optimizer, self.critic.optimizer, self.ent_coef_optimizer])
        grad_clip      = getattr(self, "max_grad_norm", 10.0)
        attn_grad_clip = getattr(self, "attn_grad_clip", 1.0)

        al, cl, ecl, ecs, qs, aent, jwm, jwstd = [], [], [], [], [], [], [], []
        log_ec = self._get_log_ent_coef()

        for _ in range(gradient_steps):
            self._n_updates += 1
            rd  = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            obs = rd.observations

            # ── Entropy coef update: standard scalar lp (NOT joint-weighted) ──
            with torch.no_grad():
                _, lp_scalar = self.actor.action_log_prob(obs)
                lp_scalar = lp_scalar.reshape(-1, 1)
            ec = torch.exp(log_ec.detach())
            ec_loss = -(log_ec * (lp_scalar + self.target_entropy).detach()).mean()
            self.ent_coef_optimizer.zero_grad(); ec_loss.backward()
            self.ent_coef_optimizer.step()
            ecl.append(ec_loss.item()); ecs.append(ec.item())

            # ── Critic update (unchanged from standard SAC) ──
            with torch.no_grad():
                na, nlp = self.actor.action_log_prob(rd.next_observations)
                nq = torch.cat(self.critic_target(rd.next_observations, na), dim=1)
                nqm, _ = torch.min(nq, dim=1, keepdim=True)
                tq = rd.rewards + (1-rd.dones)*self.gamma*(nqm - ec*nlp.reshape(-1,1))
            cq = self.critic(obs, rd.actions)
            c_loss = sum(F.mse_loss(q, tq) for q in cq)
            self.critic.optimizer.zero_grad(); c_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), grad_clip)
            self.critic.optimizer.step(); cl.append(c_loss.item())

            # ── Actor update: per-dim log_prob × 17 joint weights ──
            for p in self.critic.parameters(): p.requires_grad_(False)
            api, per_dim_lp = self.actor.action_log_prob_per_dim(obs)  # (B,A), (B,A)
            qpi = torch.cat(self.critic(obs, api), dim=1)
            mq, _ = torch.min(qpi, dim=1, keepdim=True)
            joint_weights, entropy = self.policy.joint_attn(obs)     # (B,17), scalar
            # Weighted entropy: Σ_j w_j · lp_j scaled by action_dim to restore full entropy
            # signal strength. Without *action_dim the softmax weights give ~1/17 of standard
            # entropy gradient, causing log_std to stay large → tanh saturation → α explosion.
            action_dim = per_dim_lp.shape[-1]
            weighted_lp = (joint_weights * per_dim_lp).sum(-1, keepdim=True) * action_dim  # (B,1)
            a_loss = (ec * weighted_lp - mq).mean()
            self.actor.optimizer.zero_grad(); a_loss.backward()
            actor_params = [p for n, p in self.actor.named_parameters()]
            torch.nn.utils.clip_grad_norm_(actor_params, grad_clip)
            torch.nn.utils.clip_grad_norm_(self.policy.joint_attn.parameters(), attn_grad_clip)
            self.actor.optimizer.step()
            al.append(a_loss.item()); qs.append(mq.mean().item())
            aent.append(entropy.item())
            jwm.append(joint_weights.mean().item())   # should hover near 1/17≈0.059
            jwstd.append(joint_weights.std(unbiased=False).item())
            for p in self.critic.parameters(): p.requires_grad_(True)
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.mul_(1-self.tau).add_(self.tau*p.data)

        self.logger.record("train/n_updates",     self._n_updates)
        self.logger.record("train/actor_loss",    np.mean(al))
        self.logger.record("train/critic_loss",   np.mean(cl))
        self.logger.record("train/ent_coef",      np.mean(ecs))
        self.logger.record("train/ent_coef_loss", np.mean(ecl))
        self.logger.record("train/mean_q1",       np.mean(qs))
        self.logger.record("train/attn_mean",     np.mean(jwm))
        self.logger.record("train/attn_std",      np.mean(jwstd))
        self.logger.record("train/attn_entropy",  np.mean(aent))
        self.logger.record("train/contact_align_loss", np.nan)


class ContactJointSAC(ContactJointLossMixin, SAC): pass


# ═══════════════════════════════════════════════════════════════════════════════
# Method 4b: contact_joint_score_loss
#
# Root-cause fix for Method 4's permanently-uniform attention:
#   Method 4 used SOFTMAX (Σw_j=1) → grad ∝ (lp_j − mean_lp) ≈ 0 at init.
#   Method 4b uses SIGMOID → each gate is independent ∈(0,1), total ∈(0,17).
#
# Architecture: 17 learnable joint queries cross-attend to K=16 contact tokens
# → (B,17,E) → sigmoid → (B,17) per-joint contact gates.
#
# Actor loss: (joint_scores(B,17) × advantage(B,1)).sum(-1).mean()
#   • Same formula as Method B but with 17 JOINT-aligned scores (not K token scores)
#   • Gradient for gate_j = advantage = (α×lp − Q): nonzero, diverse per sample
#   • 308-dim contact → 17-dim per-joint gates (not 1 scalar, not sum-to-1)
# ═══════════════════════════════════════════════════════════════════════════════

class JointContactScorer(nn.Module):
    """17 learnable joint queries cross-attend to K contact tokens → 17 sigmoid scores."""
    def __init__(self, contact_dim=308, action_dim=17, embed_dim=64,
                 num_heads=4, num_contact_tokens=16):
        super().__init__()
        self.num_tokens = num_contact_tokens
        self.embed_dim  = embed_dim
        self.action_dim = action_dim
        self.contact_enc = nn.Sequential(
            nn.Linear(contact_dim, 256), nn.ReLU(),
            nn.Linear(256, num_contact_tokens * embed_dim))
        self.joint_queries = nn.Embedding(action_dim, embed_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True, dropout=0.0)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.score_head = nn.Linear(embed_dim, 1)

    def forward(self, obs):
        dev = obs.device
        contact = obs[:, CONTACT_IDX.to(dev)]   # (B, 308)
        B = contact.shape[0]
        c_tokens = self.contact_enc(contact).view(B, self.num_tokens, self.embed_dim)
        joint_q  = self.joint_queries.weight.unsqueeze(0).expand(B, -1, -1)  # (B,17,E)
        attn_out, _ = self.cross_attn(joint_q, c_tokens, c_tokens)           # (B,17,E)
        attn_out = self.layer_norm(attn_out)
        scores = torch.sigmoid(self.score_head(attn_out).squeeze(-1))        # (B,17)∈(0,1)
        return scores


class ContactJointScoreSACPolicy(SACPolicy):
    """Contact: cross-attention → 17 independent sigmoid joint scores (no softmax)."""
    def __init__(self, *a, embed_dim=64, num_heads=4, num_contact_tokens=16, **kw):
        self._cjs_embed_dim  = embed_dim
        self._cjs_num_heads  = num_heads
        self._cjs_num_tokens = num_contact_tokens
        super().__init__(*a, **kw)

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        act_dim = self.action_space.shape[0]
        self.joint_scorer = JointContactScorer(
            action_dim=act_dim,
            embed_dim=self._cjs_embed_dim,
            num_heads=self._cjs_num_heads,
            num_contact_tokens=self._cjs_num_tokens).to(self.device)
        self.actor.optimizer = self.optimizer_class(
            list(self.actor.parameters()) + list(self.joint_scorer.parameters()),
            lr=lr_schedule(1), **self.optimizer_kwargs)

    def make_actor(self, fe=None):
        return KinematicsActor(
            **self._update_features_extractor(self.actor_kwargs, fe)).to(self.device)


class ContactJointScoreLossMixin(AttentionLossMixin):
    """
    actor_loss = (joint_scores(B,17) × advantage(B,1)).sum(-1).mean()

    joint_scores: 17 INDEPENDENT sigmoid gates (NOT softmax → NOT sum-to-1).
    Per-joint gate ∈ (0,1). Total per-sample weight ∈ (0,17).
    Cross-attention: 17 joint queries × K contact tokens → richer than K token scores.

    Gradient for gate_j = advantage = (α×lp − Q): always nonzero and sample-diverse
    → attention learns which joint-contact combinations correlate with high Q.
    """
    def train(self, gradient_steps, batch_size=64):
        self.policy.set_training_mode(True)
        self._update_learning_rate(
            [self.actor.optimizer, self.critic.optimizer, self.ent_coef_optimizer])
        grad_clip      = getattr(self, "max_grad_norm", 10.0)
        attn_grad_clip = getattr(self, "attn_grad_clip", 1.0)

        al, cl, ecl, ecs, qs, jsm, jsstd = [], [], [], [], [], [], []
        log_ec = self._get_log_ent_coef()

        for _ in range(gradient_steps):
            self._n_updates += 1
            rd  = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            obs = rd.observations

            # ── Standard SAC entropy coef update ──
            with torch.no_grad():
                _, lp = self.actor.action_log_prob(obs); lp = lp.reshape(-1, 1)
            ec = torch.exp(log_ec.detach())
            ec_loss = -(log_ec * (lp + self.target_entropy).detach()).mean()
            self.ent_coef_optimizer.zero_grad(); ec_loss.backward()
            self.ent_coef_optimizer.step()
            ecl.append(ec_loss.item()); ecs.append(ec.item())

            # ── Standard SAC critic update ──
            with torch.no_grad():
                na, nlp = self.actor.action_log_prob(rd.next_observations)
                nq = torch.cat(self.critic_target(rd.next_observations, na), dim=1)
                nqm, _ = torch.min(nq, dim=1, keepdim=True)
                tq = rd.rewards + (1-rd.dones)*self.gamma*(nqm - ec*nlp.reshape(-1,1))
            cq = self.critic(obs, rd.actions)
            c_loss = sum(F.mse_loss(q, tq) for q in cq)
            self.critic.optimizer.zero_grad(); c_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), grad_clip)
            self.critic.optimizer.step(); cl.append(c_loss.item())

            # ── Actor update: 17 per-joint sigmoid scores × advantage ──
            for p in self.critic.parameters(): p.requires_grad_(False)
            api, lp = self.actor.action_log_prob(obs); lp = lp.reshape(-1, 1)
            qpi = torch.cat(self.critic(obs, api), dim=1)
            mq, _ = torch.min(qpi, dim=1, keepdim=True)
            joint_scores = self.policy.joint_scorer(obs)      # (B,17) ∈ (0,1)^17
            adv = (ec * lp - mq)                              # (B,1) scalar advantage
            a_loss = (joint_scores * adv).sum(dim=-1).mean()  # sum joints, mean batch
            self.actor.optimizer.zero_grad(); a_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.actor.parameters()), grad_clip)
            torch.nn.utils.clip_grad_norm_(
                list(self.policy.joint_scorer.parameters()), attn_grad_clip)
            self.actor.optimizer.step()
            al.append(a_loss.item()); qs.append(mq.mean().item())
            jsm.append(joint_scores.mean().item())
            jsstd.append(joint_scores.std(unbiased=False).item())
            for p in self.critic.parameters(): p.requires_grad_(True)
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.mul_(1-self.tau).add_(self.tau*p.data)

        self.logger.record("train/n_updates",        self._n_updates)
        self.logger.record("train/actor_loss",        np.mean(al))
        self.logger.record("train/critic_loss",       np.mean(cl))
        self.logger.record("train/ent_coef",          np.mean(ecs))
        self.logger.record("train/ent_coef_loss",     np.mean(ecl))
        self.logger.record("train/mean_q1",           np.mean(qs))
        self.logger.record("train/attn_mean",         np.mean(jsm))
        self.logger.record("train/attn_std",          np.mean(jsstd))
        self.logger.record("train/attn_entropy",      np.nan)
        self.logger.record("train/contact_align_loss",np.nan)


class ContactJointScoreSAC(ContactJointScoreLossMixin, SAC): pass


def get_log_alpha(model):
    if not hasattr(model, "ent_coef_optimizer"): return None
    params = model.ent_coef_optimizer.param_groups[0].get("params", [])
    return params[0] if params else None


class MetricsCallback(BaseCallback):
    def __init__(self, run_dir, log_interval=2000, reward_threshold=3000.0,
                 checkpoint_freq=25000, best_model_window=50, alpha_min=0.05,
                 alpha_max=None, verbose=1):
        super().__init__(verbose)
        self.run_dir=run_dir; self.log_interval=log_interval
        self.reward_threshold=reward_threshold; self.checkpoint_freq=checkpoint_freq
        self.best_model_window=best_model_window; self.alpha_min=alpha_min
        self.log_alpha_min=float(np.log(alpha_min))
        self.log_alpha_max=float(np.log(alpha_max)) if alpha_max is not None else None
        self.timesteps=[]; self.mean_rewards=[]; self.std_rewards=[]
        self.min_rewards=[]; self.max_rewards=[]; self.ep_lengths=[]
        self.actor_losses=[]; self.critic_losses=[]; self.q_values=[]
        self.ent_coefs=[]; self.fps_history=[]; self.success_rates=[]
        self.ram_history=[]; self.attn_means=[]; self.attn_stds=[]; self.attn_entropies=[]
        self.contact_align_losses=[]
        self.ep_reward_window=deque(maxlen=100); self.ep_length_window=deque(maxlen=100)
        self.success_window=deque(maxlen=100)
        self.best_mean_reward=-np.inf; self.best_step=0
        self.last_time=time.time(); self.last_steps=0; self.log_alpha_param=None
        self.metrics_path=self.run_dir/"metrics.csv"
        self.checkpoints_dir=ensure_dir(self.run_dir/"checkpoints")
        self._write_header()

    def _write_header(self):
        with self.metrics_path.open("w",newline="",encoding="utf-8") as f:
            csv.writer(f).writerow(["step","mean_reward","std_reward","min_reward","max_reward",
                "ep_length","critic_loss","actor_loss","q_value","alpha","fps","ram_gb",
                "success_rate","attn_mean","attn_std","attn_entropy","contact_align_loss"])

    def _on_training_start(self): self.log_alpha_param = get_log_alpha(self.model)

    def _on_step(self):
        if self.log_alpha_param is not None:
            with torch.no_grad():
                self.log_alpha_param.clamp_(
                    min=self.log_alpha_min,
                    max=self.log_alpha_max if self.log_alpha_max is not None else float("inf"))
        _t = time.time()
        if not hasattr(self,"_last_hb"): self._last_hb = _t
        if _t - self._last_hb >= 2.0:
            print(".", end="", flush=True); self._last_hb = _t
        for info in self.locals.get("infos",[]):
            if "episode" in info:
                r=info["episode"]["r"]; l=info["episode"]["l"]
                self.ep_reward_window.append(r); self.ep_length_window.append(l)
                self.success_window.append(int(r>self.reward_threshold))
                print(f"  >> ep done | reward {r:>8.1f} | len {int(l):>5} | step {self.num_timesteps:>8,}", flush=True)
        if self.n_calls>0 and self.n_calls%self.checkpoint_freq==0:
            ck=self.checkpoints_dir/f"model_{self.num_timesteps}"
            self.model.save(str(ck))
            ve=self.model.get_vec_normalize_env()
            if ve is not None: ve.save(str(ck)+"_vecnorm.pkl")
        if self.n_calls%self.log_interval!=0 or not self.ep_reward_window: return True
        logs=self.model.logger.name_to_value
        mr=float(np.mean(self.ep_reward_window)); sr=float(np.std(self.ep_reward_window))
        mnr=float(np.min(self.ep_reward_window)); mxr=float(np.max(self.ep_reward_window))
        el=float(np.mean(self.ep_length_window)); sc=float(np.mean(self.success_window)*100)
        al=float(logs.get("train/actor_loss",np.nan)); cl=float(logs.get("train/critic_loss",np.nan))
        qv=float(logs.get("train/mean_q1",logs.get("train/q1_values",logs.get("train/qf1_values",np.nan))))
        alpha=float(logs.get("train/ent_coef",np.nan))
        atm=float(logs.get("train/attn_mean",np.nan)); ats=float(logs.get("train/attn_std",np.nan))
        ate=float(logs.get("train/attn_entropy",np.nan))
        cal=float(logs.get("train/contact_align_loss",np.nan))
        now=time.time(); fps=(self.num_timesteps-self.last_steps)/max(now-self.last_time,1e-6)
        self.last_time=now; self.last_steps=self.num_timesteps; ram=ram_gb()
        self.timesteps.append(self.num_timesteps); self.mean_rewards.append(mr)
        self.std_rewards.append(sr); self.min_rewards.append(mnr); self.max_rewards.append(mxr)
        self.ep_lengths.append(el); self.actor_losses.append(al); self.critic_losses.append(cl)
        self.q_values.append(qv); self.ent_coefs.append(alpha); self.fps_history.append(fps)
        self.success_rates.append(sc); self.ram_history.append(ram)
        self.attn_means.append(atm); self.attn_stds.append(ats); self.attn_entropies.append(ate)
        self.contact_align_losses.append(cal)
        with self.metrics_path.open("a",newline="",encoding="utf-8") as f:
            csv.writer(f).writerow([self.num_timesteps,mr,sr,mnr,mxr,el,cl,al,qv,alpha,fps,ram,sc,atm,ats,ate,cal])
        recent=list(self.ep_reward_window)[-self.best_model_window:]
        roll=float(np.mean(recent)) if recent else mr
        if roll>self.best_mean_reward:
            self.best_mean_reward=roll; self.best_step=self.num_timesteps
            self.model.save(str(self.run_dir/"best_model"))
            ve=self.model.get_vec_normalize_env()
            if ve is not None: ve.save(str(self.run_dir/"best_model_vecnorm.pkl"))
        if self.verbose:
            print(f"step={self.num_timesteps:>8,} reward={mr:>8.1f} best={self.best_mean_reward:>8.1f} "
                  f"fps={fps:>6.1f} alpha={alpha:>7.4f} ram={ram:>5.1f}GB", flush=True)
        return True


class CollapseDetectionCallback(BaseCallback):
    def __init__(self, metrics_callback, collapse_threshold=0.20, min_steps=50000, verbose=1):
        super().__init__(verbose)
        self.mc=metrics_callback; self.threshold=collapse_threshold
        self.min_steps=min_steps; self._peak=-np.inf; self._in_col=False; self._log=[]
    def _on_step(self):
        if self.num_timesteps<self.min_steps or not self.mc.ep_reward_window: return True
        mr=float(np.mean(self.mc.ep_reward_window))
        self._peak=max(self._peak,mr)
        if self._peak>500:
            drop=(self._peak-mr)/self._peak
            if drop>self.threshold and not self._in_col:
                self._in_col=True
                ev={"step":self.num_timesteps,"peak":round(self._peak,1),
                    "current":round(mr,1),"drop_pct":round(drop*100,1)}
                self._log.append(ev)
                ck=self.mc.run_dir/f"collapse_ckpt_{self.num_timesteps}"
                self.model.save(str(ck))
                ve=self.model.get_vec_normalize_env()
                if ve is not None: ve.save(str(ck)+"_vecnorm.pkl")
                lp=self.mc.run_dir/"collapse_log.csv"; hdr=not lp.exists()
                with lp.open("a",newline="",encoding="utf-8") as f:
                    w=csv.DictWriter(f,fieldnames=list(ev.keys()))
                    if hdr: w.writeheader()
                    w.writerow(ev)
                if self.verbose:
                    print(f"\n  [COLLAPSE] step={self.num_timesteps:,} "
                          f"{self._peak:.0f}->{mr:.0f} ({drop*100:.0f}% drop) ckpt saved", flush=True)
            elif drop<0.05: self._in_col=False
        return True
    @property
    def n_collapses(self): return len(self._log)


def record_random_video(config, output_path, n_episodes=2, max_steps_per_ep=1000):
    env=make_render_env(config["env_id"],config["seed"])
    frames,rewards,lengths=[],[],[]
    try:
        for ep in range(n_episodes):
            obs,_=env.reset(seed=config["seed"]+ep); done=False; er=0.0; el=0
            while not done and el<max_steps_per_ep:
                a=env.action_space.sample(); obs,r,te,tr,_=env.step(a)
                frames.append(env.render()); er+=r; el+=1; done=te or tr
            rewards.append(er); lengths.append(el)
    finally: env.close()
    imageio.mimsave(output_path,frames,fps=config["video_fps"])
    return {"video_type":"random","mean_reward":float(np.mean(rewards)) if rewards else None,
            "frames":len(frames),"path":str(output_path)}


def record_trained_video(model, vecnorm_path, config, output_path,
                         deterministic=True, n_episodes=2, max_steps_per_ep=1000):
    eb=DummyVecEnv([make_env(config["env_id"],config["seed"])])
    ee=VecNormalize.load(str(vecnorm_path),eb); ee.training=False; ee.norm_reward=False
    re=make_render_env(config["env_id"],config["seed"])
    frames,rewards,lengths=[],[],[]
    try:
        for ep in range(n_episodes):
            obs=ee.reset(); re.reset(seed=config["seed"]+ep); done=False; er=0.0; el=0
            while not done and el<max_steps_per_ep:
                a,_=model.predict(obs,deterministic=deterministic)
                obs,_,da,_=ee.step(a); _,r,te,tr,_=re.step(a[0])
                frames.append(re.render()); er+=r; el+=1; done=bool(da[0]) or te or tr
            rewards.append(er); lengths.append(el)
    finally: ee.close(); re.close()
    imageio.mimsave(output_path,frames,fps=config["video_fps"])
    return {"video_type":"trained","mean_reward":float(np.mean(rewards)) if rewards else None,
            "frames":len(frames),"path":str(output_path)}


def build_model(experiment, train_env, config, device):
    common=dict(env=train_env,learning_rate=config["learning_rate"],
        buffer_size=config["buffer_size"],learning_starts=config["learning_starts"],
        batch_size=config["batch_size"],tau=config["tau"],gamma=config["gamma"],
        optimize_memory_usage=config["optimize_memory_usage"],
        replay_buffer_kwargs=dict(handle_timeout_termination=False),
        gradient_steps=config["gradient_steps"],train_freq=config["train_freq"],
        ent_coef=f"auto_{config['ent_coef_init']}",
        target_entropy=-float(train_env.action_space.shape[0]),
        seed=config["seed"],verbose=0,device=device,
        tensorboard_log=str(config["run_dir"]/"tb_logs"))
    ak=dict(net_arch=config["net_arch"],activation_fn=torch.nn.ReLU,
            optimizer_kwargs=dict(eps=1e-5),embed_dim=config["embed_dim"],
            num_heads=config["num_heads"],contact_hidden=config["contact_hidden"])
    if experiment=="vanilla_sac":
        m=SAC(policy="MlpPolicy",policy_kwargs=dict(net_arch=config["net_arch"]),**common)
    elif experiment=="attentive_sac":
        m=AttentiveSAC(policy=AttentiveSACPolicy,policy_kwargs=ak,**common)
    elif experiment=="kin_only_no_attn":
        m=SAC(policy=KinematicsOnlySACPolicy,
              policy_kwargs=dict(net_arch=config["net_arch"],activation_fn=torch.nn.ReLU,
                                 optimizer_kwargs=dict(eps=1e-5)),**common)
    elif experiment=="full_obs_attn_loss":
        m=FullObsAttentionLossSAC(policy=FullObsAttentionLossPolicy,policy_kwargs=ak,**common)
    elif experiment=="kin_root_attentive":
        m=AttentiveSAC(policy=KinRootAttentiveSACPolicy,policy_kwargs=ak,**common)
    elif experiment=="centered_attentive":
        m=AttentiveSAC(policy=CenteredAttentiveSACPolicy,policy_kwargs=ak,**common)
    elif experiment=="normalized_attentive":
        m=AttentiveSAC(policy=NormalizedAttentiveSACPolicy,policy_kwargs=ak,**common)
    elif experiment=="kin_root_normalized_attentive":
        m=AttentiveSAC(policy=KinRootNormalizedAttentiveSACPolicy,policy_kwargs=ak,**common)
    elif experiment=="film_attentive":
        m=FiLMSAC(policy=FiLMSACPolicy,
                  policy_kwargs=dict(net_arch=config["net_arch"],activation_fn=torch.nn.ReLU,
                                     optimizer_kwargs=dict(eps=1e-5),film_dim=config["embed_dim"]),
                  **common)
    elif experiment=="stable_attentive":
        m=AttentiveSAC(policy=AttentiveSACPolicy,
                       policy_kwargs={**ak,"weight_mode":"sigmoid2x"},**common)
        m.attn_grad_clip=config.get("attn_grad_clip",1.0)
    elif experiment=="stable_kin_root":
        m=AttentiveSAC(policy=KinRootNormalizedAttentiveSACPolicy,
                       policy_kwargs={**ak,"weight_mode":"normalized_exp","attn_scale":0.25},**common)
        m.attn_grad_clip=config.get("attn_grad_clip",1.0)
    elif experiment=="rich_contact_attn":
        ak_rich=dict(net_arch=config["net_arch"],activation_fn=torch.nn.ReLU,
                     optimizer_kwargs=dict(eps=1e-5),
                     embed_dim=config["embed_dim"],num_heads=config["num_heads"],
                     contact_hidden=config["contact_hidden"],
                     num_contact_tokens=config.get("num_contact_tokens",8),attn_scale=0.5)
        m=RichContactAttentiveSAC(policy=RichContactAttentiveSACPolicy,policy_kwargs=ak_rich,**common)
        m.attn_grad_clip=config.get("attn_grad_clip",1.0)
    elif experiment=="contact_additive_loss":
        ak_ca=dict(net_arch=config["net_arch"],activation_fn=torch.nn.ReLU,
                   optimizer_kwargs=dict(eps=1e-5),
                   embed_dim=config["embed_dim"],num_heads=config["num_heads"],
                   contact_hidden=config["contact_hidden"],
                   num_tokens=config.get("num_contact_tokens",8))
        m=ContactAdditiveSAC(policy=ContactAdditiveSACPolicy,policy_kwargs=ak_ca,**common)
        m.attn_grad_clip=config.get("attn_grad_clip",1.0)
        m.contact_align_lambda=config.get("contact_align_lambda",0.1)
    elif experiment=="contact_per_token_loss":
        ak_ct=dict(net_arch=config["net_arch"],activation_fn=torch.nn.ReLU,
                   optimizer_kwargs=dict(eps=1e-5),
                   num_contact_tokens=config.get("num_contact_tokens",8),
                   contact_token_hidden=128)
        m=ContactTokenSAC(policy=ContactTokenSACPolicy,policy_kwargs=ak_ct,**common)
        m.attn_grad_clip=config.get("attn_grad_clip",1.0)
    elif experiment=="contact_joint_weight_loss":
        ak_cj=dict(net_arch=config["net_arch"],activation_fn=torch.nn.ReLU,
                   optimizer_kwargs=dict(eps=1e-5),
                   embed_dim=config["embed_dim"],num_heads=config["num_heads"],
                   num_contact_tokens=config.get("num_contact_tokens",16))
        m=ContactJointSAC(policy=ContactJointWeightSACPolicy,policy_kwargs=ak_cj,**common)
        m.attn_grad_clip=config.get("attn_grad_clip",1.0)
    elif experiment=="contact_joint_score_loss":
        ak_cjs=dict(net_arch=config["net_arch"],activation_fn=torch.nn.ReLU,
                    optimizer_kwargs=dict(eps=1e-5),
                    embed_dim=config["embed_dim"],num_heads=config["num_heads"],
                    num_contact_tokens=config.get("num_contact_tokens",16))
        m=ContactJointScoreSAC(policy=ContactJointScoreSACPolicy,policy_kwargs=ak_cjs,**common)
        m.attn_grad_clip=config.get("attn_grad_clip",1.0)
    else:
        raise ValueError(f"Unknown experiment: {experiment!r}")
    if hasattr(m,"max_grad_norm"): m.max_grad_norm=config["max_grad_norm"]
    return m


def count_total_params(model): return sum(p.numel() for p in model.policy.parameters())


def save_summary(run_dir, model, callback, config):
    s={"experiment":config["experiment"],"seed":config["seed"],
       "peak_reward":float(np.nanmax(callback.mean_rewards)) if callback.mean_rewards else None,
       "reward_at_500k":interpolate_metric_at_step(callback.timesteps,callback.mean_rewards,500_000),
       "reward_at_1M":interpolate_metric_at_step(callback.timesteps,callback.mean_rewards,1_000_000),
       "mean_fps":float(np.nanmean(callback.fps_history)) if callback.fps_history else None,
       "total_params":int(count_total_params(model)),"best_step":int(callback.best_step),
       "best_mean_reward":float(callback.best_mean_reward) if np.isfinite(callback.best_mean_reward) else None,
       "peak_ram_gb":float(np.nanmax(callback.ram_history)) if callback.ram_history else ram_gb(),
       "final_step":int(callback.timesteps[-1]) if callback.timesteps else 0}
    (run_dir/"summary.json").write_text(json.dumps(s,indent=2),encoding="utf-8")
    return s


def plot_curves(run_dir, callback, config):
    if not callback.timesteps: return
    x=np.array(callback.timesteps)/1000.0
    fig,axes=plt.subplots(2,3,figsize=(16,9))
    fig.suptitle(f"{config['experiment']} | seed {config['seed']}",fontsize=14)
    axes[0,0].plot(x,callback.mean_rewards,color="#0f766e")
    axes[0,0].fill_between(x,np.array(callback.mean_rewards)-np.array(callback.std_rewards),
        np.array(callback.mean_rewards)+np.array(callback.std_rewards),color="#5eead4",alpha=0.3)
    axes[0,0].set_title("Reward"); axes[0,0].set_xlabel("Timesteps (k)")
    axes[0,1].plot(x,callback.actor_losses,label="actor",color="#dc2626")
    axes[0,1].plot(x,callback.critic_losses,label="critic",color="#2563eb")
    axes[0,1].set_title("Losses"); axes[0,1].set_xlabel("Timesteps (k)"); axes[0,1].legend()
    axes[0,2].plot(x,callback.q_values,color="#7c3aed")
    axes[0,2].set_title("Mean Q"); axes[0,2].set_xlabel("Timesteps (k)")
    axes[1,0].plot(x,callback.ent_coefs,color="#ea580c")
    axes[1,0].set_title("Entropy Coef"); axes[1,0].set_xlabel("Timesteps (k)")
    axes[1,1].plot(x,callback.fps_history,color="#1d4ed8")
    axes[1,1].set_title("FPS"); axes[1,1].set_xlabel("Timesteps (k)")
    if any(np.isfinite(v) for v in callback.attn_means):
        axes[1,2].plot(x,callback.attn_means,label="mean",color="#059669")
        axes[1,2].plot(x,callback.attn_stds,label="std",color="#a16207")
        axes[1,2].legend(); axes[1,2].set_title("Attention Stats")
    else:
        axes[1,2].plot(x,callback.success_rates,color="#0891b2")
        axes[1,2].set_title("Success Rate")
    axes[1,2].set_xlabel("Timesteps (k)")
    for ax in axes.flat: ax.grid(True,alpha=0.2)
    fig.tight_layout(); fig.savefig(run_dir/"curves.png",bbox_inches="tight"); plt.close(fig)


def plot_all_metrics(run_dir, callback, config):
    if not callback.timesteps: return
    x=np.array(callback.timesteps)/1000.0
    series=[("Mean Reward",callback.mean_rewards,"#0f766e"),
            ("Std Reward",callback.std_rewards,"#14b8a6"),
            ("Min Reward",callback.min_rewards,"#ef4444"),
            ("Max Reward",callback.max_rewards,"#22c55e"),
            ("Episode Length",callback.ep_lengths,"#2563eb"),
            ("Actor Loss",callback.actor_losses,"#dc2626"),
            ("Critic Loss",callback.critic_losses,"#7c3aed"),
            ("Mean Q",callback.q_values,"#8b5cf6"),
            ("Entropy Coef",callback.ent_coefs,"#ea580c"),
            ("FPS",callback.fps_history,"#1d4ed8"),
            ("RAM (GB)",callback.ram_history,"#475569"),
            ("Success Rate",callback.success_rates,"#0891b2"),
            ("Attention Mean",callback.attn_means,"#059669"),
            ("Attention Std",callback.attn_stds,"#a16207"),
            ("Attention Entropy",callback.attn_entropies,"#7e22ce"),
            ("Contact Align Loss",callback.contact_align_losses,"#b45309")]
    fig,axes=plt.subplots(4,4,figsize=(20,16))
    fig.suptitle(f"All Metrics | {config['experiment']} | seed {config['seed']}",fontsize=15)
    for ax,(title,values,color) in zip(axes.flat,series):
        arr=np.array(values,dtype=float)
        if np.isfinite(arr).any(): ax.plot(x,arr,color=color,linewidth=2)
        ax.set_title(title,fontsize=10); ax.set_xlabel("Timesteps (k)"); ax.grid(True,alpha=0.2)
    for ax in axes.flat[len(series):]: ax.axis("off")
    fig.tight_layout(); fig.savefig(run_dir/"all_metrics.png",bbox_inches="tight",dpi=120); plt.close(fig)


def update_comparison_table(results_root):
    rows=[]
    for sp in results_root.glob("*/summary.json"):
        try: rows.append(json.loads(sp.read_text(encoding="utf-8")))
        except json.JSONDecodeError: continue
    if not rows: return
    fn=sorted({k for r in rows for k in r})
    with (results_root/"comparison_table.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fn); w.writeheader(); w.writerows(rows)


def default_run_name(config):
    exp=config["experiment"]; seed=config["seed"]
    names={"attentive_sac":f"attentive_sac_seed{seed}",
           "vanilla_sac":f"vanilla_sac_seed{seed}",
           "kin_only_no_attn":f"ablation_kin_only_no_attn_seed{seed}",
           "full_obs_attn_loss":f"ablation_full_obs_attn_loss_seed{seed}",
           "kin_root_attentive":f"refine_kin_root_attentive_seed{seed}",
           "centered_attentive":f"refine_centered_attentive_seed{seed}",
           "normalized_attentive":f"refine_normalized_attentive_seed{seed}",
           "kin_root_normalized_attentive":f"refine_kin_root_normalized_attentive_seed{seed}",
           "film_attentive":f"film_attentive_seed{seed}",
           "stable_attentive":f"stable_attentive_seed{seed}",
           "stable_kin_root":f"stable_kin_root_seed{seed}",
           "rich_contact_attn":f"rich_contact_attn_seed{seed}",
           "contact_additive_loss":f"contact_additive_loss_seed{seed}",
           "contact_per_token_loss":f"contact_per_token_loss_seed{seed}",
           "contact_joint_weight_loss":f"contact_joint_weight_loss_seed{seed}",
           "contact_joint_score_loss":f"contact_joint_score_loss_seed{seed}"}
    return names.get(exp,f"{exp}_seed{seed}")


def run_experiment(config):
    from stable_baselines3.common.callbacks import CallbackList
    results_root=ensure_dir(Path(config["results_root"]))
    run_dir=ensure_dir(results_root/config["run_name"])
    config["run_dir"]=run_dir
    (run_dir/"config.json").write_text(json.dumps(config,indent=2,default=str),encoding="utf-8")
    before=record_random_video(config,run_dir/"before_training.mp4",
        n_episodes=config["video_episodes"],max_steps_per_ep=config["video_max_steps"])
    (run_dir/"before_video.json").write_text(json.dumps(before,indent=2),encoding="utf-8")
    train_env=build_vec_env(config)
    model=build_model(config["experiment"],train_env,config,config["device"])
    cb=MetricsCallback(run_dir=run_dir,log_interval=config["log_interval"],
        reward_threshold=config["reward_threshold"],checkpoint_freq=config["checkpoint_freq"],
        best_model_window=config["best_model_window"],alpha_min=config["alpha_min"],
        alpha_max=config.get("alpha_max",None),verbose=1)
    ccb=CollapseDetectionCallback(cb,collapse_threshold=0.20,min_steps=50000,verbose=1)
    start=time.time()
    try:
        model.learn(total_timesteps=config["total_timesteps"],
                    callback=CallbackList([cb,ccb]),log_interval=1,progress_bar=False)
    except Exception as exc:
        (run_dir/"crash_log.txt").write_text("\n".join([
            f"experiment: {config['experiment']}",f"run_name: {config['run_name']}",
            f"timesteps_logged: {cb.timesteps[-1] if cb.timesteps else 0}",
            f"error_type: {type(exc).__name__}",f"error_message: {exc}"]),encoding="utf-8")
        raise
    finally:
        elapsed=time.time()-start
        model.save(str(run_dir/"final_model")); train_env.save(str(run_dir/"final_vecnorm.pkl"))
        config["elapsed_seconds"]=elapsed; config["n_collapses"]=ccb.n_collapses
        (run_dir/"config.json").write_text(json.dumps(config,indent=2,default=str),encoding="utf-8")
        train_env.close()
    summary=save_summary(run_dir,model,cb,config)
    summary["n_collapses"]=ccb.n_collapses; summary["attn_grad_clip"]=config.get("attn_grad_clip")
    (run_dir/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    plot_curves(run_dir,cb,config); plot_all_metrics(run_dir,cb,config)
    vn=run_dir/"best_model_vecnorm.pkl"; mp=run_dir/"best_model.zip"
    if mp.exists() and vn.exists():
        after=record_trained_video(model,vn,config,run_dir/"after_training.mp4",
            n_episodes=config["video_episodes"],max_steps_per_ep=config["video_max_steps"])
        (run_dir/"after_video.json").write_text(json.dumps(after,indent=2),encoding="utf-8")
    update_comparison_table(results_root)
    return summary


def build_parser():
    p=argparse.ArgumentParser(description="Humanoid-v4 research runner")
    p.add_argument("--experiment",default="attentive_sac",choices=[
        "attentive_sac","vanilla_sac","kin_only_no_attn","full_obs_attn_loss",
        "kin_root_attentive","centered_attentive","normalized_attentive",
        "kin_root_normalized_attentive","film_attentive","stable_attentive","stable_kin_root",
        "rich_contact_attn","contact_additive_loss","contact_per_token_loss",
        "contact_joint_weight_loss","contact_joint_score_loss"])
    p.add_argument("--run-name",default=None)
    p.add_argument("--results-root",default=r"S:\rl_humanoid_runs\results")
    p.add_argument("--env-id",default="Humanoid-v4")
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--total-timesteps",type=int,default=100000)
    p.add_argument("--learning-rate",type=float,default=3e-4)
    p.add_argument("--buffer-size",type=int,default=500000)
    p.add_argument("--batch-size",type=int,default=256)
    p.add_argument("--learning-starts",type=int,default=10000)
    p.add_argument("--tau",type=float,default=0.005)
    p.add_argument("--gamma",type=float,default=0.99)
    p.add_argument("--gradient-steps",type=int,default=1)
    p.add_argument("--train-freq",type=int,default=1)
    p.add_argument("--max-grad-norm",type=float,default=10.0)
    p.add_argument("--alpha-min",type=float,default=0.05)
    p.add_argument("--alpha-max",type=float,default=None,
                   help="Upper bound for entropy coefficient (default: None = unbounded)")
    p.add_argument("--attn-grad-clip",type=float,default=1.0,
                   help="Gradient clip for attention weighter only (default 1.0, tighter than actor)")
    p.add_argument("--ent-coef-init",type=float,default=0.1)
    p.add_argument("--clip-obs",type=float,default=5.0)
    p.add_argument("--clip-reward",type=float,default=10.0)
    p.add_argument("--log-interval",type=int,default=5000)
    p.add_argument("--checkpoint-freq",type=int,default=25000)
    p.add_argument("--best-model-window",type=int,default=50)
    p.add_argument("--reward-threshold",type=float,default=3000.0)
    p.add_argument("--optimize-memory-usage",action="store_true")
    p.add_argument("--embed-dim",type=int,default=64)
    p.add_argument("--num-heads",type=int,default=4)
    p.add_argument("--contact-hidden",type=int,default=256)
    p.add_argument("--num-contact-tokens",type=int,default=8,
                   help="Number of contact tokens for RichContactAttention (default 8)")
    p.add_argument("--contact-align-lambda",type=float,default=0.1,
                   help="Weight for additive contact alignment loss (default 0.1)")
    p.add_argument("--device",default="auto")
    p.add_argument("--video-fps",type=int,default=30)
    p.add_argument("--video-episodes",type=int,default=2)
    p.add_argument("--video-max-steps",type=int,default=1000)
    return p


def main():
    parser=build_parser(); args=parser.parse_args(); config=vars(args)
    config["device"]=get_device(config["device"]); config["net_arch"]=[512,512]
    config["run_name"]=config.get("run_name") or default_run_name(config)
    print(f"device={config['device']}  torch={torch.__version__}  cuda={torch.cuda.is_available()}")
    print(f"env={config['env_id']}  experiment={config['experiment']}  run={config['run_name']}")
    print(f"results_root={config['results_root']}")
    print(f"obs split  root={len(ROOT_IDX)}  kin={len(KIN_IDX)}  contact={len(CONTACT_IDX)}")
    print(f"attn_grad_clip={config['attn_grad_clip']}  alpha_min={config['alpha_min']}  alpha_max={config.get('alpha_max')}")
    print(json.dumps(run_experiment(config), indent=2))


if __name__ == "__main__":
    main()
