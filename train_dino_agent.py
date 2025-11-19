# train_dino_agent.py
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", module="pygame.pkgdata")
warnings.filterwarnings("ignore", message=".*pkg_resources.*deprecated.*")
warnings.filterwarnings("ignore", message=".*TensorFlow installation not found.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*pkg_resources.*")

import argparse
import logging
import math
import os
import time
from dataclasses import dataclass, field
from collections import deque
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv
from gymnasium.wrappers import RecordVideo
from torch.utils.tensorboard import SummaryWriter

from utilities.helpers import make_env      # must pass through phase kwargs
from utilities.observations import set_seed, ObsNorm
from chrome_dino_env import ChromeDinoEnv   # only used for evaluation video runs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("train_dino")


# ----------------------------
# PPO model
# ----------------------------
class PpoModel(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.policy = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, act_dim),
        )
        self.value = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.policy(x)
        value = self.value(x).squeeze(-1)
        return logits, value

    @torch.no_grad()
    def act(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(x)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        logp = dist.log_prob(action)
        return action, logp, value

    def eval_actions(self, x: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(x)
        dist = torch.distributions.Categorical(logits=logits)
        logp = dist.log_prob(actions)
        entropy = dist.entropy()
        return logp, entropy, value


# ----------------------------
# Config (includes curriculum)
# ----------------------------
@dataclass
class PPOConfig:
    # PPO core
    seed: int = 42
    total_timesteps: int = 10_000_000
    n_envs: int = 16
    n_steps: int = 1024
    gamma: float = 0.995
    gae_lambda: float = 0.95
    update_epochs: int = 4
    clip_coef: float = 0.2
    vf_clip_coef: float = 0.2
    ent_coef: float = 0.02
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    lr: float = 2.5e-4
    linear_lr: bool = True
    hidden: int = 512
    torch_compile: bool = False
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    norm_obs: bool = True
    minibatch_size: int = 1024  # tune to your GPU; mb <= n_envs*n_steps

    # Logging / I/O
    log_dir: str = "./pt_logs"
    save_path: str = "./dino_ppo.pt"
    vec_backend: str = "async"  # "sync" or "async"

    # Default single-phase fallbacks (used if auto_curriculum=False)
    frame_skip: int = 1
    speed_increases: bool = False
    alive_reward: float = 0.05
    death_penalty: float = -1.0
    avoid_reward: float = 0.0
    milestone_points: int = 0
    milestone_bonus: float = 0.0
    obs_speed_cap: float = 100.0
    obs_ttc_cap: float = 300.0

    # Curriculum toggles
    auto_curriculum: bool = True

    # Phase schedule as fractions of total_timesteps (monotonic)
    phase_breaks: Tuple[float, ...] = (0.0, 0.05, 0.50)

    # Kwargs per phase
    phase_kwargs: Tuple[Dict, ...] = field(default_factory=lambda: (
    # Phase 0: birds only (teach duck)
    dict(
        frame_skip=1,
        speed_increases=False,
        spawn_probs=(0.0, 0.0, 1.0),
        duck_window_ttc=(6, 24),
        duck_bonus=0.40,
        wrong_jump_penalty=0.25,
        idle_duck_penalty=0.01,
        airtime_penalty=0.006,
        alive_reward=0.03,
        death_penalty=-1.0,
        avoid_reward=0.0,
        milestone_points=0,
        milestone_bonus=0.0,
        obs_speed_cap=100.0,
        obs_ttc_cap=300.0,
    ),
    # Phase 1: mix, fixed speed
    dict(
        frame_skip=1,
        speed_increases=False,
        spawn_probs=(0.30, 0.20, 0.50),
        duck_window_ttc=(6, 24),
        duck_bonus=0.20,
        wrong_jump_penalty=0.10,
        idle_duck_penalty=0.01,
        airtime_penalty=0.004,
        alive_reward=0.03,
        death_penalty=-1.0,
        avoid_reward=0.01, 
        milestone_points=0,
        milestone_bonus=0.0,
        obs_speed_cap=100.0,
        obs_ttc_cap=300.0,
    ),
    # Phase 2: mix + speed increases
    dict(
        frame_skip=1,
        speed_increases=True,
        spawn_probs=(0.40, 0.20, 0.40),
        duck_window_ttc=(6, 24),
        duck_bonus=0.10,
        wrong_jump_penalty=0.05,
        idle_duck_penalty=0.005,
        airtime_penalty=0.003,
        alive_reward=0.02,
        death_penalty=-1.0,
        avoid_reward=0.01,   # <--- changed
        milestone_points=0,
        milestone_bonus=0.0,
        obs_speed_cap=100.0,
        obs_ttc_cap=300.0,
    ),
))


    # ------- Skill-gated promotion knobs (phase 0 -> 1) -------
    skill_window_episodes: int = 400
    min_duck_rate_p0: float = 0.20           # >=20% of actions are duck in Phase 0
    min_avoid_avg_p0: float = 1.0            # average avoided per episode (Phase 0)

    # Optional early promotion Phase 1 -> 2, based on performance
    min_return_avg_p1: float = 50.0          # average ep return threshold
    min_duck_rate_p1: float = 0.08           # still demonstrate ducking sometimes


# ----------------------------
# Training
# ----------------------------
def ppo_train(cfg: PPOConfig):
    set_seed(cfg.seed)
    os.makedirs(cfg.log_dir, exist_ok=True)
    tb = SummaryWriter(log_dir=cfg.log_dir)

    # ---------- Video setup ----------
    run_id = time.strftime("%Y%m%d-%H%M%S")
    video_dir = os.path.join(cfg.log_dir, "videos", run_id)
    milestones = {int(cfg.total_timesteps * p): int(p * 100) for p in (0.10, 0.30, 0.60, 1.00)}
    recorded_steps = set()

    # ---------- Build curriculum thresholds ----------
    if cfg.auto_curriculum:
        assert len(cfg.phase_breaks) == len(cfg.phase_kwargs), "phase_breaks and phase_kwargs lengths must match"
        thresholds: List[Tuple[int, Dict]] = [
            (int(cfg.total_timesteps * frac), kwargs)
            for frac, kwargs in zip(cfg.phase_breaks, cfg.phase_kwargs)
        ]
    else:
        thresholds = [(0, dict(
            frame_skip=cfg.frame_skip,
            speed_increases=cfg.speed_increases,
            spawn_probs=(0.30, 0.20, 0.50),
            duck_window_ttc=(6, 24),
            duck_bonus=0.0,
            wrong_jump_penalty=0.0,
            idle_duck_penalty=0.0,
            airtime_penalty=0.0,
            alive_reward=cfg.alive_reward,
            death_penalty=cfg.death_penalty,
            avoid_reward=cfg.avoid_reward,
            milestone_points=cfg.milestone_points,
            milestone_bonus=cfg.milestone_bonus,
            obs_speed_cap=cfg.obs_speed_cap,
            obs_ttc_cap=cfg.obs_ttc_cap,
        ))]

    phase_idx = 0
    phase_kwargs = thresholds[phase_idx][1].copy()

    # ---------- Env builders ----------
    def build_envs(phase_kws: Dict):
        if cfg.vec_backend == "async" and cfg.n_envs > 1:
            return AsyncVectorEnv([make_env(i, cfg.seed, **phase_kws) for i in range(cfg.n_envs)])
        else:
            return SyncVectorEnv([make_env(i, cfg.seed, **phase_kws) for i in range(cfg.n_envs)])

    envs = build_envs(phase_kwargs)
    next_obs, _ = envs.reset(seed=cfg.seed)
    obs_dim = next_obs.shape[1]
    act_dim = envs.single_action_space.n

    # ---------- Obs norm ----------
    obs_norm = ObsNorm(obs_dim) if cfg.norm_obs else None
    if obs_norm is not None:
        obs_norm.update(next_obs)
        next_obs = obs_norm.normalize(next_obs)

    # ---------- Model / opt ----------
    model = PpoModel(obs_dim, act_dim, hidden=cfg.hidden).to(cfg.device)
    if cfg.torch_compile:
        model = torch.compile(model)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    # ---------- Storage ----------
    steps_per_update = cfg.n_steps
    batch = cfg.n_envs * steps_per_update

    obs_buf = np.zeros((steps_per_update, cfg.n_envs, obs_dim), dtype=np.float32)
    act_buf = np.zeros((steps_per_update, cfg.n_envs), dtype=np.int64)
    logp_buf = np.zeros((steps_per_update, cfg.n_envs), dtype=np.float32)
    rew_buf = np.zeros((steps_per_update, cfg.n_envs), dtype=np.float32)
    done_buf = np.zeros((steps_per_update, cfg.n_envs), dtype=np.float32)
    val_buf = np.zeros((steps_per_update, cfg.n_envs), dtype=np.float32)

    # ---------- Skill-gating bookkeeping ----------
    per_env_duck_cnt = np.zeros(cfg.n_envs, dtype=np.int64)
    per_env_ep_len   = np.zeros(cfg.n_envs, dtype=np.int64)

    duck_rate_hist = deque(maxlen=cfg.skill_window_episodes)
    avoided_hist   = deque(maxlen=cfg.skill_window_episodes)
    return_hist    = deque(maxlen=cfg.skill_window_episodes)

    def maybe_switch_phase(global_step: int):
        nonlocal envs, next_obs, phase_idx, phase_kwargs
        target_idx = phase_idx
        for i, (th, _) in enumerate(thresholds):
            if global_step >= th:
                target_idx = i

        # skill-gated 0 -> 1
        if cfg.auto_curriculum and phase_idx == 0 and len(duck_rate_hist) >= max(20, cfg.n_envs):
            avg_duck = float(np.mean(duck_rate_hist))
            avg_avoid = float(np.mean(avoided_hist)) if avoided_hist else 0.0
            if avg_duck >= cfg.min_duck_rate_p0 and avg_avoid >= cfg.min_avoid_avg_p0:
                target_idx = max(target_idx, 1)

        # skill-gated 1 -> 2
        if cfg.auto_curriculum and phase_idx == 1 and len(return_hist) >= max(20, cfg.n_envs):
            avg_ret = float(np.mean(return_hist))
            avg_duck = float(np.mean(duck_rate_hist)) if duck_rate_hist else 0.0
            if avg_ret >= cfg.min_return_avg_p1 and avg_duck >= cfg.min_duck_rate_p1:
                target_idx = max(target_idx, 2)

        if target_idx != phase_idx:
            phase_idx = target_idx
            phase_kwargs = thresholds[phase_idx][1].copy()
            logger.info(f"[{global_step}] Switching to phase {phase_idx} with kwargs={phase_kwargs}")
            envs.close()
            envs = build_envs(phase_kwargs)
            next_obs, _ = envs.reset(seed=cfg.seed)
            per_env_duck_cnt.fill(0)
            per_env_ep_len.fill(0)
            if obs_norm is not None:
                obs_norm.update(next_obs)
                next_obs = obs_norm.normalize(next_obs)

    # ---------- Video helper ----------
    @torch.no_grad()
    def record_policy_video(tag_percent: int, max_steps: int = 5000):
        fname = f"at_{tag_percent}pct"
        logger.info(f"Recording video at {tag_percent}%...")
        eval_env = ChromeDinoEnv(render_mode="rgb_array", seed=cfg.seed, **phase_kwargs)
        rec_env = RecordVideo(eval_env, video_folder=video_dir, name_prefix=fname)
        obs, _ = rec_env.reset(seed=cfg.seed)
        done = False
        steps = 0
        while not done and steps < max_steps:
            if obs_norm is not None:
                obs = obs_norm.normalize(obs)
            x = torch.as_tensor(obs, dtype=torch.float32, device=cfg.device).unsqueeze(0)
            logits, _ = model(x)
            action = int(torch.argmax(logits, dim=-1).item())
            obs, _, term, trunc, _ = rec_env.step(action)
            done = bool(term or trunc)
            steps += 1
        rec_env.close()
        logger.info(f"Saved video: {os.path.join(video_dir, fname)}.*")

    # ---------- Main loop ----------
    global_step = 0
    updates = math.ceil(cfg.total_timesteps / batch)
    start_time = time.time()
    episode_counter = 0

    for update in range(1, updates + 1):
        if cfg.linear_lr:
            frac = 1.0 - (update - 1) / updates
            for pg in optimizer.param_groups:
                pg["lr"] = cfg.lr * frac

        action_counts = np.zeros(act_dim, dtype=np.int64)
        # Simple entropy decay: from cfg.ent_coef to ~10% of it
        ent_frac = 0.1 + 0.9 * (1.0 - (update - 1) / updates)
        current_ent_coef = cfg.ent_coef * ent_frac

        # ------- Collect rollout -------
        for t in range(steps_per_update):
            global_step += cfg.n_envs
            obs_buf[t] = next_obs

            maybe_switch_phase(global_step)

            # Milestone videos
            for ms_step, pct in milestones.items():
                if global_step >= ms_step and ms_step not in recorded_steps:
                    try:
                        record_policy_video(pct)
                    except Exception as e:
                        logger.warning(f"Video recording failed at {pct}% ({ms_step} steps): {e}")
                    recorded_steps.add(ms_step)

            # Policy step
            obs_t = torch.as_tensor(next_obs, device=cfg.device, dtype=torch.float32)
            with torch.no_grad():
                action_t, logp_t, value_t = model.act(obs_t)
            action_np = action_t.cpu().numpy()
            logp_np = logp_t.cpu().numpy()
            value_np = value_t.cpu().numpy()

            # Env step
            next_obs, rewards, terms, truncs, infos = envs.step(action_np)
            dones = np.logical_or(terms, truncs).astype(np.float32)

            # Store
            act_buf[t] = action_np
            logp_buf[t] = logp_np
            val_buf[t] = value_np
            rew_buf[t] = rewards
            done_buf[t] = dones

            per_env_ep_len += 1
            per_env_duck_cnt += (action_np == 2).astype(np.int64)

            for a in action_np:
                action_counts[int(a)] += 1

            # Final episode infos (auto-resets)
            final_infos = infos.get("final_info", None)
            if final_infos is not None:
                for env_i, fi in enumerate(final_infos):
                    if fi is not None and "episode" in fi:
                        ep_r = float(np.asarray(fi["episode"]["r"]).item())
                        ep_l = int(np.asarray(fi["episode"]["l"]).item())
                        avoided = int(fi.get("avoided_count", 0))
                        avoided_bird = int(fi.get("avoided_bird", 0))
                        avoided_other = int(fi.get("avoided_other", max(0, avoided - avoided_bird)))
                        avoid_r_tot = float(fi.get("avoid_reward_total", 0.0))

                        ducks = int(per_env_duck_cnt[env_i])
                        drate = ducks / max(1, ep_l)
                        duck_rate_hist.append(drate)
                        avoided_hist.append(avoided)
                        return_hist.append(ep_r)

                        per_env_duck_cnt[env_i] = 0
                        per_env_ep_len[env_i] = 0

                        episode_counter += 1
                        logger.info(
                            f"[{global_step}] ep={episode_counter} "
                            f"return={ep_r:8.1f} len={ep_l:5d} "
                            f"avoided={avoided:3d} (bird={avoided_bird}, other={avoided_other}) "
                            f"duck_rate={drate:.3f} avoid_R={avoid_r_tot:.4f}"
                        )

                        # TB episode scalars
                        tb.add_scalar("episode/return", ep_r, episode_counter)
                        tb.add_scalar("episode/length", ep_l, episode_counter)
                        tb.add_scalar("episode/avoided", avoided, episode_counter)
                        tb.add_scalar("episode/avoided_bird", avoided_bird, episode_counter)
                        tb.add_scalar("episode/avoided_other", avoided_other, episode_counter)
                        tb.add_scalar("episode/avoid_reward_total", avoid_r_tot, episode_counter)
                        tb.add_scalar("episode/duck_rate", drate, episode_counter)
                        tb.add_scalar("episode/return_vs_step", ep_r, global_step)
                        tb.add_scalar("episode/length_vs_step", ep_l, global_step)

            if obs_norm is not None:
                obs_norm.update(next_obs)
                next_obs = obs_norm.normalize(next_obs)

        # ------- Compute advantages (GAE-λ) -------
        with torch.no_grad():
            next_obs_t = torch.as_tensor(next_obs, device=cfg.device, dtype=torch.float32)
            _, next_val_t = model.forward(next_obs_t)
            next_val = next_val_t.cpu().numpy()

        adv_buf = np.zeros_like(rew_buf)
        lastgaelam = np.zeros((cfg.n_envs,), dtype=np.float32)
        for t in reversed(range(steps_per_update)):
            not_done = 1.0 - done_buf[t]
            delta = rew_buf[t] + cfg.gamma * next_val * not_done - val_buf[t]
            lastgaelam = delta + cfg.gamma * cfg.gae_lambda * not_done * lastgaelam
            adv_buf[t] = lastgaelam
            next_val = val_buf[t]
        ret_buf = adv_buf + val_buf

        # ------- Flatten and update -------
        b_obs = torch.as_tensor(obs_buf.reshape(batch, obs_dim), device=cfg.device)
        b_act = torch.as_tensor(act_buf.reshape(batch), device=cfg.device)
        b_logp_old = torch.as_tensor(logp_buf.reshape(batch), device=cfg.device)
        b_adv = torch.as_tensor(adv_buf.reshape(batch), device=cfg.device)
        b_ret = torch.as_tensor(ret_buf.reshape(batch), device=cfg.device)
        b_val_old = torch.as_tensor(val_buf.reshape(batch), device=cfg.device)

        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        inds = np.arange(batch)
        mb = cfg.minibatch_size

        pol_losses, val_losses, ents = [], [], []

        for epoch in range(cfg.update_epochs):
            np.random.shuffle(inds)
            for start in range(0, batch, mb):
                end = start + mb
                mb_inds = inds[start:end]

                logits, value = model.forward(b_obs[mb_inds])
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(b_act[mb_inds])
                entropy = dist.entropy().mean()

                ratio = torch.exp(logp - b_logp_old[mb_inds])
                surr1 = ratio * b_adv[mb_inds]
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef) * b_adv[mb_inds]
                policy_loss = -torch.min(surr1, surr2).mean()

                if cfg.vf_clip_coef > 0:
                    v_clipped = b_val_old[mb_inds] + (value - b_val_old[mb_inds]).clamp(-cfg.vf_clip_coef, cfg.vf_clip_coef)
                    v_loss_unclipped = (value - b_ret[mb_inds]) ** 2
                    v_loss_clipped = (v_clipped - b_ret[mb_inds]) ** 2
                    value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    value_loss = 0.5 * (b_ret[mb_inds] - value).pow(2).mean()

                loss = policy_loss + cfg.vf_coef * value_loss - current_ent_coef * entropy


                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()

                pol_losses.append(policy_loss.item())
                val_losses.append(value_loss.item())
                ents.append(entropy.item())

        # ------- Train logs -------
        if pol_losses:
            tb.add_scalar("train/policy_loss", float(np.mean(pol_losses)), global_step)
            tb.add_scalar("train/value_loss", float(np.mean(val_losses)), global_step)
            tb.add_scalar("train/entropy", float(np.mean(ents)), global_step)
            tb.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
            tb.add_scalar("rollout/reward_mean", rew_buf.mean(), global_step)
            tb.add_scalar("rollout/return_mean", ret_buf.mean(), global_step)

            total_actions = action_counts.sum()
            if total_actions > 0:
                for a in range(len(action_counts)):
                    tb.add_scalar(f"actions/freq_{a}", action_counts[a] / total_actions, global_step)

        # ------- Checkpoint -------
        if update % 10 == 0 or update == updates:
            to_save = model._orig_mod if hasattr(model, "_orig_mod") else model
            payload = {
                "model_state_dict": to_save.state_dict(),
                "cfg": vars(cfg),
                "obs_norm": None if obs_norm is None else {
                    "mean": obs_norm.mean.tolist(),
                    "var": obs_norm.var.tolist(),
                    "count": float(obs_norm.count),
                    "clip": float(obs_norm.clip),
                },
            }
            torch.save(payload, cfg.save_path)
            elapsed = time.time() - start_time
            logger.info(f"[update {update}/{updates}] saved -> {cfg.save_path} | elapsed={elapsed/60:.1f} min")
            tb.add_scalar("time/elapsed_minutes", elapsed / 60.0, global_step)
            tb.add_scalar("time/steps_per_second", global_step / max(1.0, elapsed), global_step)

    envs.close()
    tb.close()

    if not recorded_steps:
        try:
            record_policy_video(100)
        except Exception as e:
            logger.warning(f"Final video recording failed at 100%: {e}")

    logger.info("Training done.")


# ----------------------------
# CLI
# ----------------------------
def parse_args() -> PPOConfig:
    d = PPOConfig()
    p = argparse.ArgumentParser()
    # Core
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--total-timesteps", type=int, default=d.total_timesteps)
    p.add_argument("--n-envs", type=int, default=d.n_envs)
    p.add_argument("--n-steps", type=int, default=d.n_steps)
    p.add_argument("--update-epochs", type=int, default=d.update_epochs)
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--hidden", type=int, default=d.hidden)
    p.add_argument("--clip-coef", type=float, default=d.clip_coef)
    p.add_argument("--vf-clip-coef", type=float, default=d.vf_clip_coef)
    p.add_argument("--ent-coef", type=float, default=d.ent_coef)
    p.add_argument("--vf-coef", type=float, default=d.vf_coef)
    p.add_argument("--gamma", type=float, default=d.gamma)
    p.add_argument("--gae-lambda", type=float, default=d.gae_lambda)
    p.add_argument("--max-grad-norm", type=float, default=d.max_grad_norm)
    p.add_argument("--torch-compile", action="store_true")
    p.add_argument("--no-linear-lr", action="store_true")
    p.add_argument("--no-norm-obs", action="store_true")
    p.add_argument("--vec-backend", choices=["sync", "async"], default=d.vec_backend)
    p.add_argument("--save-path", type=str, default=d.save_path)
    p.add_argument("--log-dir", type=str, default=d.log_dir)
    p.add_argument("--minibatch-size", type=int, default=d.minibatch_size)

    # Curriculum toggle
    p.add_argument("--no-auto-curriculum", action="store_true")

    # Single-phase fallback knobs (only used if no-auto-curriculum)
    p.add_argument("--frame-skip", type=int, default=d.frame_skip)
    p.add_argument("--speed-increases", action="store_true", default=d.speed_increases)
    p.add_argument("--alive-reward", type=float, default=d.alive_reward)
    p.add_argument("--death-penalty", type=float, default=d.death_penalty)
    p.add_argument("--avoid-reward", type=float, default=d.avoid_reward)
    p.add_argument("--milestone-points", type=int, default=d.milestone_points)
    p.add_argument("--milestone-bonus", type=float, default=d.milestone_bonus)
    p.add_argument("--obs-speed-cap", type=float, default=d.obs_speed_cap)
    p.add_argument("--obs-ttc-cap", type=float, default=d.obs_ttc_cap)

    a = p.parse_args()
    return PPOConfig(
        seed=a.seed,
        total_timesteps=a.total_timesteps,
        n_envs=a.n_envs,
        n_steps=a.n_steps,
        update_epochs=a.update_epochs,
        lr=a.lr,
        hidden=a.hidden,
        clip_coef=a.clip_coef,
        vf_clip_coef=a.vf_clip_coef,
        ent_coef=a.ent_coef,
        vf_coef=a.vf_coef,
        gamma=a.gamma,
        gae_lambda=a.gae_lambda,
        max_grad_norm=a.max_grad_norm,
        torch_compile=a.torch_compile,
        linear_lr=not a.no_linear_lr,
        norm_obs=not a.no_norm_obs,
        vec_backend=a.vec_backend,
        save_path=a.save_path,
        log_dir=a.log_dir,
        minibatch_size=a.minibatch_size,
        auto_curriculum=not a.no_auto_curriculum,
        frame_skip=a.frame_skip,
        speed_increases=a.speed_increases,
        alive_reward=a.alive_reward,
        death_penalty=a.death_penalty,
        avoid_reward=a.avoid_reward,
        milestone_points=a.milestone_points,
        milestone_bonus=a.milestone_bonus,
        obs_speed_cap=a.obs_speed_cap,
        obs_ttc_cap=a.obs_ttc_cap,
    )


if __name__ == "__main__":
    logger.info("Starting training…")
    cfg = parse_args()
    logger.info(f"Config: {cfg}")
    ppo_train(cfg)
