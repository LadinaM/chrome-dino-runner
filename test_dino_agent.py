import argparse
import numpy as np
import torch

from torch.serialization import add_safe_globals

from chrome_dino_env import ChromeDinoEnv
from train_dino_agent import PPO_Model, ObsNorm


def load_checkpoint(path: str, obs_dim: int, act_dim: int, device: str = "cpu"):
    # 1) Load (trusted local ckpt)

    add_safe_globals([np.core.multiarray._reconstruct])  # keep if weights_only=True path is used anywhere
    data = torch.load(path, map_location=device, weights_only=False)

    hidden = data.get("cfg", {}).get("hidden", 128)
    model = PPO_Model(obs_dim, act_dim, hidden=hidden).to(device)

    # 2) Handle torch.compile state dicts
    state_dict = data["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {k.replace("_orig_mod.", "", 1): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()

    # 3) Rebuild obs_norm if present (supports lists or numpy)
    obs_norm = None
    on = data.get("obs_norm")
    if on is not None:
        obs_norm = ObsNorm(obs_dim)
        obs_norm.mean = np.array(on["mean"], dtype=np.float64)
        obs_norm.var  = np.array(on["var"],  dtype=np.float64)
        obs_norm.count = float(on["count"])
        obs_norm.clip  = float(on["clip"])

    return model, obs_norm, data.get("cfg", {})



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="./dino_ppo.pt")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    # Human render env
    env = ChromeDinoEnv(render_mode="human", seed=123)
    obs, _ = env.reset()
    obs_dim = obs.shape[-1]
    act_dim = env.action_space.n

    model, obs_norm, cfg = load_checkpoint(args.model, obs_dim, act_dim, device=args.device)

    for ep in range(args.episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            if obs_norm is not None:
                obs = obs_norm.normalize(obs)

            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=args.device).unsqueeze(0)
            with torch.no_grad():
                logits, _ = model.forward(obs_t)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.probs.argmax(dim=-1)  # greedy
                action = int(action.item())

            obs, r, terminated, truncated, info = env.step(action)
            ep_ret += float(r)
            done = terminated or truncated

        print(f"[Episode {ep+1}] return={ep_ret:.1f} score={info.get('score', 'n/a')}")
    env.close()


if __name__ == "__main__":
    main()
