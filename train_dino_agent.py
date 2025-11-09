from typing import Literal
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
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv
from gymnasium.wrappers import RecordVideo
from torch.utils.tensorboard import SummaryWriter

from utilities.helpers import make_env
from utilities.observations import set_seed, ObsNorm
from chrome_dino_env import ChromeDinoEnv  # only used for video eval

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


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

    def forward(self, x):
        logits = self.policy(x)
        value = self.value(x).squeeze(-1)
        return logits, value

    def act(self, x):
        logits, value = self.forward(x)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        logp = dist.log_prob(action)
        return action, logp, value

    def eval_actions(self, x, actions):
        logits, value = self.forward(x)
        dist = torch.distributions.Categorical(logits=logits)
        logp = dist.log_prob(actions)
        entropy = dist.entropy()
        return logp, entropy, value



@dataclass
class PPOConfig:
    seed: int = 42
    total_timesteps: int = 6_000_000
    n_envs: int = 8
    n_steps: int = 2048         # per env
    gamma: float = 0.99
    gae_lambda: float = 0.95
    update_epochs: int = 5
    clip_coef: float = 0.2
    vf_clip_coef: float = 0.1
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    lr: float = 2.5e-4
    linear_lr: bool = True
    hidden: int = 512
    torch_compile: bool = False
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    norm_obs: bool = True
    log_dir: str = "./pt_logs"
    save_path: str = "./dino_ppo.pt"
    vec_backend: str = "async"  # "sync" or "async"
    minibatch_size: int = 256

    # Base env rewards/dynamics (fallbacks for non-curriculum mode)
    speed_increases: bool = False
    alive_reward: float = 1.0
    death_penalty: float = -100.0
    avoid_reward: float = 20.0
    milestone_points: int = 10
    milestone_bonus: float = 20.0

    # Auto-curriculum toggle
    auto_curriculum: bool = True


def ppo_train(cfg: PPOConfig):
    set_seed(cfg.seed)
    os.makedirs(cfg.log_dir, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=cfg.log_dir)

    # Unique run-specific video directory to avoid overwrite warnings
    run_id = time.strftime("%Y%m%d-%H%M%S")
    video_dir = os.path.join(cfg.log_dir, "videos", run_id)

    # Record videos at % of total timesteps: 10%, 30%, 60%, 100%
    milestone_percents = [0.10, 0.30, 0.60, 1.00]
    video_step_milestones = {int(cfg.total_timesteps * p): int(p * 100) for p in milestone_percents}
    recorded_step_milestones = set()

    # ---- Automatic curriculum phases (by % of total timesteps)
    # Phase 0: birds only, fixed speed (teach duck)
    # Phase 1: mix in cacti, still fixed speed
    # Phase 2: mix + speed increases on
    phase_schedule = [
        (0.00, {  # 0% – 30%
            "frame_skip": 1,
            "speed_increases": False,
            "spawn_probs": (0.0, 0.0, 1.0),  # birds only
            "bird_only_phase": True,
            # shaping
            "duck_window_ttc": (6, 24),
            "duck_bonus": 0.3,
            "wrong_jump_penalty": 0.2,
            "idle_duck_penalty": 0.01,
            "airtime_penalty": 0.005,
            # obs caps
            "obs_speed_cap": 100.0,
            "obs_ttc_cap": 300.0,
            # base rewards (gentle)
            "alive_reward": 0.05,
            "death_penalty": -1.0,
            "avoid_reward": 0.0,
            "milestone_points": 0,
            "milestone_bonus": 0.0,
        }),
        (0.1, {  # 10% – 60%
            "frame_skip": 1,
            "speed_increases": False,
            "spawn_probs": (0.3, 0.2, 0.5),  # mix in cacti
            "bird_only_phase": False,
            "duck_window_ttc": (6, 24),
            "duck_bonus": 0.25,
            "wrong_jump_penalty": 0.15,
            "idle_duck_penalty": 0.01,
            "airtime_penalty": 0.004,
            "alive_reward": 0.05,
            "death_penalty": -1.0,
            "avoid_reward": 0.0,
            "milestone_points": 0,
            "milestone_bonus": 0.0,
        }),
        (0.6, {  # 60% – 100%
            "frame_skip": 1,
            "speed_increases": True,        # turn on speed progression
            "spawn_probs": (0.4, 0.2, 0.4),
            "bird_only_phase": False,
            "duck_window_ttc": (6, 24),
            "duck_bonus": 0.2,              # decay shaping
            "wrong_jump_penalty": 0.1,
            "idle_duck_penalty": 0.005,
            "airtime_penalty": 0.003,
            "alive_reward": 0.05,
            "death_penalty": -1.0,
            "avoid_reward": 0.0,
            "milestone_points": 0,
            "milestone_bonus": 0.0,
        }),
    ]
    phase_thresholds = [(int(cfg.total_timesteps * frac), kwargs) for frac, kwargs in phase_schedule]
    current_phase_idx = 0
    current_phase_kwargs = phase_thresholds[current_phase_idx][1]

    # If auto-curriculum is disabled, build a single-phase kwargs from cfg
    if not cfg.auto_curriculum:
        current_phase_kwargs = dict[str, int | tuple[float, float, float] | tuple[Literal[6], Literal[24]] | float](
            frame_skip=1,
            speed_increases=cfg.speed_increases,
            spawn_probs=(0.3, 0.2, 0.5),
            bird_only_phase=False,
            duck_window_ttc=(6, 24),
            duck_bonus=0.0,
            wrong_jump_penalty=0.0,
            idle_duck_penalty=0.0,
            airtime_penalty=0.0,
            obs_speed_cap=100.0,
            obs_ttc_cap=300.0,
            alive_reward=cfg.alive_reward,
            death_penalty=cfg.death_penalty,
            avoid_reward=cfg.avoid_reward,
            milestone_points=cfg.milestone_points,
            milestone_bonus=cfg.milestone_bonus,
        )

    # Vector env builder
    def build_envs(phase_kwargs):
        if cfg.vec_backend == "async" and cfg.n_envs > 1:
            return AsyncVectorEnv([make_env(i, cfg.seed, **phase_kwargs) for i in range(cfg.n_envs)])
        else:
            return SyncVectorEnv([make_env(i, cfg.seed, **phase_kwargs) for i in range(cfg.n_envs)])

    # Initial envs
    envs = build_envs(current_phase_kwargs)
    next_obs, _ = envs.reset(seed=cfg.seed)
    obs_dim = next_obs.shape[1]
    act_dim = envs.single_action_space.n

    # Optional obs normalization
    obs_norm = ObsNorm(obs_dim) if cfg.norm_obs else None
    if obs_norm is not None:
        obs_norm.update(next_obs)
        next_obs = obs_norm.normalize(next_obs)

    # ---- Model
    model = PpoModel(obs_dim, act_dim, hidden=cfg.hidden).to(cfg.device)
    if cfg.torch_compile:
        model = torch.compile(model)

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    start_time = time.time()

    # Storage
    num_steps = cfg.n_steps
    batch_size = cfg.n_envs * num_steps

    obs_buf = np.zeros((num_steps, cfg.n_envs, obs_dim), dtype=np.float32)
    act_buf = np.zeros((num_steps, cfg.n_envs), dtype=np.int64)
    logp_buf = np.zeros((num_steps, cfg.n_envs), dtype=np.float32)
    rew_buf = np.zeros((num_steps, cfg.n_envs), dtype=np.float32)
    done_buf = np.zeros((num_steps, cfg.n_envs), dtype=np.float32)
    val_buf = np.zeros((num_steps, cfg.n_envs), dtype=np.float32)

    global_step = 0
    updates = math.ceil(cfg.total_timesteps / batch_size)
    episode_count = 0

    # Video helper uses the *current phase* kwargs so eval matches training conditions
    def record_policy_video(tag_percent: int, max_steps: int = 5000):
        fname = f"at_{tag_percent}pct"
        logger.info(f"Recording video at {tag_percent}%...")
        eval_env = ChromeDinoEnv(render_mode="rgb_array", seed=cfg.seed, **current_phase_kwargs)
        rec_env = RecordVideo(eval_env, video_folder=video_dir, name_prefix=fname)
        obs, _ = rec_env.reset(seed=cfg.seed)
        done = False
        steps = 0
        while not done and steps < max_steps:
            if obs_norm is not None:
                obs = obs_norm.normalize(obs)
            with torch.no_grad():
                x = torch.as_tensor(obs, dtype=torch.float32, device=cfg.device).unsqueeze(0)
                logits, _ = model(x)
                action = int(torch.argmax(logits, dim=-1).item())  # deterministic for clarity
            obs, _, term, trunc, _ = rec_env.step(action)
            done = bool(term or trunc)
            steps += 1
        rec_env.close()
        logger.info(f"Saved video: {os.path.join(video_dir, fname)}.*")

    for update in range(1, updates + 1):
        # LR schedule
        if cfg.linear_lr:
            frac = 1.0 - (update - 1) / updates
            for pg in optimizer.param_groups:
                pg["lr"] = cfg.lr * frac

        # rollout action histogram (debug)
        action_counts = np.zeros(act_dim, dtype=np.int64)

        # Collect rollout
        for step in range(num_steps):
            global_step += cfg.n_envs
            obs_buf[step] = next_obs

            # ---- Auto-curriculum phase switching ----
            if cfg.auto_curriculum:
                next_phase_idx = current_phase_idx
                for i, (th_step, kwargs) in enumerate(phase_thresholds):
                    if global_step >= th_step:
                        next_phase_idx = i
                if next_phase_idx != current_phase_idx:
                    current_phase_idx = next_phase_idx
                    current_phase_kwargs = phase_thresholds[current_phase_idx][1]
                    logger.info(f"Switching to phase {current_phase_idx} at step {global_step} with kwargs={current_phase_kwargs}")
                    envs.close()
                    envs = build_envs(current_phase_kwargs)
                    next_obs, _ = envs.reset(seed=cfg.seed)
                    if obs_norm is not None:
                        obs_norm.update(next_obs)
                        next_obs = obs_norm.normalize(next_obs)

            # Check step-based video milestones (relative to total timesteps)
            for ms_step, ms_pct in video_step_milestones.items():
                if global_step >= ms_step and ms_step not in recorded_step_milestones:
                    try:
                        record_policy_video(ms_pct)
                    except Exception as e:
                        logger.warning(f"Video recording failed at {ms_pct}% ({ms_step} steps): {e}")
                    else:
                        recorded_step_milestones.add(ms_step)

            # torch forward on device
            obs_t = torch.as_tensor(next_obs, device=cfg.device, dtype=torch.float32)
            with torch.no_grad():
                action_t, logp_t, value_t = model.act(obs_t)

            action_np = action_t.cpu().numpy()
            logp_np = logp_t.cpu().numpy()
            value_np = value_t.cpu().numpy()

            # Step envs
            next_obs, rewards, terminations, truncations, infos = envs.step(action_np)
            dones = np.logical_or(terminations, truncations).astype(np.float32)

            # Store current transition
            act_buf[step] = action_np
            logp_buf[step] = logp_np
            val_buf[step] = value_np
            rew_buf[step] = rewards
            done_buf[step] = dones

            # Track action distribution
            for a in action_np:
                action_counts[int(a)] += 1

            # Episode logging via final_info (vector auto-resets)
            final_infos = infos.get("final_info", None)
            if final_infos is not None:
                for fi in final_infos:
                    if fi is not None and "episode" in fi:
                        ep_r = float(np.asarray(fi["episode"]["r"]).item())
                        ep_l = int(np.asarray(fi["episode"]["l"]).item())
                        avoided = int(fi.get("avoided_count", 0))
                        logger.info(f"[{global_step}] ep_return={ep_r:10.1f}  ep_len={ep_l:>5}  avoided={avoided}")

                        episode_count += 1
                        tb_writer.add_scalar("episode/return", ep_r, episode_count)
                        tb_writer.add_scalar("episode/length", ep_l, episode_count)
                        tb_writer.add_scalar("episode/obstacles_avoided", avoided, episode_count)
                        tb_writer.add_scalar("episode/return_vs_step", ep_r, global_step)
                        tb_writer.add_scalar("episode/length_vs_step", ep_l, global_step)
                        tb_writer.add_scalar("episode/obstacles_avoided_vs_step", avoided, global_step)

            if obs_norm is not None:
                obs_norm.update(next_obs)
                next_obs = obs_norm.normalize(next_obs)

        # Compute advantages with GAE(λ)
        with torch.no_grad():
            next_obs_t = torch.as_tensor(next_obs, device=cfg.device, dtype=torch.float32)
            _, next_value_t = model.forward(next_obs_t)
            next_value = next_value_t.cpu().numpy()

        adv_buf = np.zeros_like(rew_buf)
        lastgaelam = np.zeros((cfg.n_envs,), dtype=np.float32)
        for t in reversed(range(num_steps)):
            not_done = 1.0 - done_buf[t]
            delta = rew_buf[t] + cfg.gamma * next_value * not_done - val_buf[t]
            lastgaelam = delta + cfg.gamma * cfg.gae_lambda * not_done * lastgaelam
            adv_buf[t] = lastgaelam
            next_value = val_buf[t]

        ret_buf = adv_buf + val_buf

        # Flatten rollout
        b_obs = torch.as_tensor(obs_buf.reshape(batch_size, obs_dim), device=cfg.device)
        b_actions = torch.as_tensor(act_buf.reshape(batch_size), device=cfg.device)
        b_logp_old = torch.as_tensor(logp_buf.reshape(batch_size), device=cfg.device)
        b_adv = torch.as_tensor(adv_buf.reshape(batch_size), device=cfg.device)
        b_ret = torch.as_tensor(ret_buf.reshape(batch_size), device=cfg.device)
        b_val_old = torch.as_tensor(val_buf.reshape(batch_size), device=cfg.device).detach()

        # Advantage norm
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        # Update policy for several epochs
        inds = np.arange(batch_size)
        mb = cfg.minibatch_size

        epoch_policy_losses = []
        epoch_value_losses = []
        epoch_entropies = []

        for epoch in range(cfg.update_epochs):
            np.random.shuffle(inds)
            for start in range(0, batch_size, mb):
                end = start + mb
                mb_inds = inds[start:end]

                logits, value = model.forward(b_obs[mb_inds])
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(b_actions[mb_inds])
                entropy = dist.entropy().mean()

                # Ratio for clipped surrogate
                ratio = torch.exp(logp - b_logp_old[mb_inds])
                surr1 = ratio * b_adv[mb_inds]
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef) * b_adv[mb_inds]
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (with optional clip)
                if cfg.vf_clip_coef > 0:
                    v_clipped = b_val_old[mb_inds] + (value - b_val_old[mb_inds]).clamp(-cfg.vf_clip_coef, cfg.vf_clip_coef)
                    v_loss_unclipped = (value - b_ret[mb_inds]) ** 2
                    v_loss_clipped = (v_clipped - b_ret[mb_inds]) ** 2
                    value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    value_loss = 0.5 * (b_ret[mb_inds] - value).pow(2).mean()

                loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()

                epoch_policy_losses.append(policy_loss.item())
                epoch_value_losses.append(value_loss.item())
                epoch_entropies.append(entropy.item())

        # Log training metrics to TensorBoard
        if epoch_policy_losses:
            avg_policy_loss = np.mean(epoch_policy_losses)
            avg_value_loss = np.mean(epoch_value_losses)
            avg_entropy = np.mean(epoch_entropies)
            current_lr = optimizer.param_groups[0]["lr"]

            tb_writer.add_scalar("train/policy_loss", avg_policy_loss, global_step)
            tb_writer.add_scalar("train/value_loss", avg_value_loss, global_step)
            tb_writer.add_scalar("train/entropy", avg_entropy, global_step)
            tb_writer.add_scalar("train/learning_rate", current_lr, global_step)
            tb_writer.add_scalar("train/update", update, global_step)

            # Advantage / value / rollout stats
            tb_writer.add_scalar("train/advantage_mean", b_adv.mean().item(), global_step)
            tb_writer.add_scalar("train/advantage_std", b_adv.std().item(), global_step)
            tb_writer.add_scalar("train/value_mean", b_val_old.mean().item(), global_step)
            tb_writer.add_scalar("rollout/reward_mean", rew_buf.mean(), global_step)
            tb_writer.add_scalar("rollout/reward_sum", rew_buf.sum(), global_step)
            tb_writer.add_scalar("rollout/return_mean", ret_buf.mean(), global_step)

            # Action distribution over the rollout
            action_total = action_counts.sum()
            if action_total > 0:
                for a in range(len(action_counts)):
                    tb_writer.add_scalar(f"actions/freq_action_{a}",
                                         action_counts[a] / action_total, global_step)

        # simple checkpointing
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
            logger.info(f"[update {update}/{updates}] saved to {cfg.save_path} | elapsed={elapsed/60:.1f} min")

            tb_writer.add_scalar("time/elapsed_minutes", elapsed / 60, global_step)
            tb_writer.add_scalar("time/steps_per_second", global_step / elapsed, global_step)

    envs.close()
    tb_writer.close()

    # If training ended before any milestone was reached, record one video at 100%
    if len(recorded_step_milestones) == 0:
        try:
            record_policy_video(100)
        except Exception as e:
            logger.warning(f"Final video recording failed at 100%: {e}")

    logger.info("Training done.")


# ----------------------------
# CLI
# ----------------------------
def parse_args() -> PPOConfig:
    defaults = PPOConfig()
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=defaults.seed)
    p.add_argument("--total-timesteps", type=int, default=defaults.total_timesteps)
    p.add_argument("--n-envs", type=int, default=defaults.n_envs)
    p.add_argument("--n-steps", type=int, default=defaults.n_steps)
    p.add_argument("--update-epochs", type=int, default=defaults.update_epochs)
    p.add_argument("--lr", type=float, default=defaults.lr)
    p.add_argument("--hidden", type=int, default=defaults.hidden)
    p.add_argument("--clip-coef", type=float, default=defaults.clip_coef)
    p.add_argument("--vf-clip-coef", type=float, default=defaults.vf_clip_coef)
    p.add_argument("--ent-coef", type=float, default=defaults.ent_coef)
    p.add_argument("--vf-coef", type=float, default=defaults.vf_coef)
    p.add_argument("--gamma", type=float, default=defaults.gamma)
    p.add_argument("--gae-lambda", type=float, default=defaults.gae_lambda)
    p.add_argument("--max-grad-norm", type=float, default=defaults.max_grad_norm)
    p.add_argument("--torch-compile", action="store_true")
    p.add_argument("--no-linear-lr", action="store_true")
    p.add_argument("--no-norm-obs", action="store_true")
    p.add_argument("--vec-backend", choices=["sync", "async"], default=defaults.vec_backend)
    p.add_argument("--save-path", type=str, default=defaults.save_path)
    p.add_argument("--log-dir", type=str, default=defaults.log_dir)
    p.add_argument("--minibatch-size", type=int, default=defaults.minibatch_size)
    p.add_argument("--no-auto-curriculum", action="store_true", help="Disable automatic phase schedule")
    # fallback single-phase (only used when --no-auto-curriculum)
    p.add_argument("--alive-reward", type=float, default=defaults.alive_reward)
    p.add_argument("--death-penalty", type=float, default=defaults.death_penalty)
    p.add_argument("--avoid-reward", type=float, default=defaults.avoid_reward)
    p.add_argument("--milestone-points", type=int, default=defaults.milestone_points)
    p.add_argument("--milestone-bonus", type=float, default=defaults.milestone_bonus)

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
        # single-phase fallbacks
        alive_reward=a.alive_reward,
        death_penalty=a.death_penalty,
        avoid_reward=a.avoid_reward,
        milestone_points=a.milestone_points,
        milestone_bonus=a.milestone_bonus,
    )


if __name__ == "__main__":
    logger.info("Starting training...")
    cfg = parse_args()
    logger.info(f"Configuration: {cfg}")
    ppo_train(cfg)
