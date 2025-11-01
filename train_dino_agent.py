import argparse
import math
import os
import time
import warnings
from dataclasses import dataclass

from utilities.helpers import make_env
from utilities.observations import set_seed, ObsNorm

warnings.filterwarnings("ignore", module="pygame.pkgdata")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv


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
    total_timesteps: int = 1_000_000
    n_envs: int = 4
    n_steps: int = 2048         # per env
    gamma: float = 0.99
    gae_lambda: float = 0.95
    update_epochs: int = 10
    clip_coef: float = 0.2
    vf_clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    lr: float = 3e-4
    linear_lr: bool = True
    hidden: int = 128
    torch_compile: bool = False
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    norm_obs: bool = True
    log_dir: str = "./pt_logs"
    save_path: str = "./dino_ppo.pt"
    vec_backend: str = "async"  # "sync" or "async"
    minibatch_size: int = 64


def ppo_train(cfg: PPOConfig):
    set_seed(cfg.seed)
    os.makedirs(cfg.log_dir, exist_ok=True)

    # ---- Vectorized envs (Gymnasium API)
    if cfg.vec_backend == "async" and cfg.n_envs > 1:
        envs = AsyncVectorEnv([make_env(i, cfg.seed) for i in range(cfg.n_envs)])
    else:
        envs = SyncVectorEnv([make_env(i, cfg.seed) for i in range(cfg.n_envs)])

    # Reset: Gymnasium vector returns (obs, infos)
    next_obs, _ = envs.reset(seed=cfg.seed)
    # Shapes: obs -> (n_envs, obs_dim)
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

    next_done = np.zeros((cfg.n_envs,), dtype=np.float32)

    global_step = 0
    updates = math.ceil(cfg.total_timesteps / batch_size)

    for update in range(1, updates + 1):
        # LR schedule
        if cfg.linear_lr:
            frac = 1.0 - (update - 1) / updates
            for pg in optimizer.param_groups:
                pg["lr"] = cfg.lr * frac

        # Collect rollout
        for step in range(num_steps):
            global_step += cfg.n_envs
            obs_buf[step] = next_obs

            # torch forward on device
            obs_t = torch.as_tensor(next_obs, device=cfg.device, dtype=torch.float32)
            with torch.no_grad():
                action_t, logp_t, value_t = model.act(obs_t)
            action_np = action_t.cpu().numpy()
            logp_np = logp_t.cpu().numpy()
            value_np = value_t.cpu().numpy()

            act_buf[step] = action_np
            logp_buf[step] = logp_np
            val_buf[step] = value_np
            done_buf[step] = next_done

            # Vector step (Gymnasium vector API)
            next_obs, rewards, terminations, truncations, infos = envs.step(action_np)
            dones = np.logical_or(terminations, truncations).astype(np.float32)
            rew_buf[step] = rewards

            # Episode logging via final_info
            # Gymnasium vector auto-resets and provides final stats in infos["final_info"]
            final_infos = infos.get("final_info", None)
            if final_infos is not None:
                for fi in final_infos:
                    if fi is not None and "episode" in fi:
                        ep_r = float(np.asarray(fi["episode"]["r"]).item())
                        ep_l = int(np.asarray(fi["episode"]["l"]).item())
                        print(f"[{global_step}] ep_return={ep_r:.1f} ep_len={ep_l}")

            if obs_norm is not None:
                obs_norm.update(next_obs)
                next_obs = obs_norm.normalize(next_obs)

            next_done = dones

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
        b_val_old = torch.as_tensor(val_buf.reshape(batch_size), device=cfg.device)

        # Advantage norm
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        # Update policy for several epochs
        inds = np.arange(batch_size)
        mb = cfg.minibatch_size
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
                    v_loss_unclipped = (value - b_ret[mb_inds])**2
                    v_loss_clipped = (v_clipped - b_ret[mb_inds])**2
                    value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    value_loss = 0.5 * (b_ret[mb_inds] - value).pow(2).mean()

                loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()

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
            print(f"[update {update}/{updates}] saved to {cfg.save_path} | elapsed={elapsed/60:.1f} min")
    envs.close()
    print("Training done.")


# ----------------------------
# CLI
# ----------------------------
def parse_args() -> PPOConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--total-timesteps", type=int, default=1_000_000)
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--n-steps", type=int, default=2048)
    p.add_argument("--update-epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--clip-coef", type=float, default=0.2)
    p.add_argument("--vf-clip-coef", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--torch-compile", action="store_true")
    p.add_argument("--no-linear-lr", action="store_true")
    p.add_argument("--no-norm-obs", action="store_true")
    p.add_argument("--vec-backend", choices=["sync", "async"], default="async")
    p.add_argument("--save-path", type=str, default="./dino_ppo.pt")
    p.add_argument("--log-dir", type=str, default="./pt_logs")
    p.add_argument("--minibatch-size", type=int, default=64)

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
    )


if __name__ == "__main__":
    cfg = parse_args()
    ppo_train(cfg)
