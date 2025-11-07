import warnings
warnings.filterwarnings("ignore", module="pygame.pkgdata")
warnings.filterwarnings("ignore", message=".*pkg_resources.*deprecated.*")
warnings.filterwarnings("ignore", message=".*TensorFlow installation not found.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*pkg_resources.*")

import time, torch, numpy as np
from chrome_dino_env import ChromeDinoEnv
from train_dino_agent import PpoModel
from utilities.observations import ObsNorm
from torch.serialization import add_safe_globals

def load_ckpt(path, obs_dim, act_dim, device):
    add_safe_globals([np._core.multiarray._reconstruct])
    data = torch.load(path, map_location=device, weights_only=False)
    hidden = data.get("cfg", {}).get("hidden", 512)
    model = PpoModel(obs_dim, act_dim, hidden=hidden).to(device).eval()

    sd = data["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in sd):  # torch.compile case
        sd = {k.replace("_orig_mod.", "", 1): v for k, v in sd.items()}
    model.load_state_dict(sd)

    obs_norm = None
    if data.get("obs_norm"):
        on = data["obs_norm"]
        obs_norm = ObsNorm(obs_dim)
        obs_norm.mean = np.array(on["mean"], dtype=np.float64)
        obs_norm.var  = np.array(on["var"],  dtype=np.float64)
        obs_norm.count = float(on["count"])
        obs_norm.clip  = float(on["clip"])
    cfg = data.get("cfg", {})
    return model, obs_norm, cfg

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # build a temp env to get dims
    tmp = ChromeDinoEnv(render_mode=None, frame_skip=1)
    obs, _ = tmp.reset(); obs_dim = obs.shape[-1]; act_dim = tmp.action_space.n
    tmp.close()

    model, obs_norm, cfg = load_ckpt("./dino_ppo.pt", obs_dim, act_dim, device)

    # mirror key training knobs
    env = ChromeDinoEnv(
        render_mode="human",
        frame_skip=1,
        speed_increases=bool(cfg.get("speed_increases", False)),
        alive_reward=cfg.get("alive_reward", 0.1),
        death_penalty=cfg.get("death_penalty", -1.0),
        avoid_reward=cfg.get("avoid_reward", 1.0),
        milestone_points=cfg.get("milestone_points", 0),
        milestone_bonus=cfg.get("milestone_bonus", 0.0),
        seed=123,
    )

    for ep in range(3):
        obs, _ = env.reset()
        done = False
        while not done:
            if obs_norm is not None:
                obs = obs_norm.normalize(obs)
            with torch.no_grad():
                x = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                logits, _ = model(x)
                action = int(torch.argmax(logits, dim=-1).item())  # deterministic
            obs, _, term, trunc, _ = env.step(action)
            done = term or trunc
            # optional: throttle if your CPU/GPU is too fast
            # time.sleep(1/60)
    env.close()

if __name__ == "__main__":
    main()
