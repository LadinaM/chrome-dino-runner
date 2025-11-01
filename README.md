<p align="center">
  <img src="https://github.com/dhhruv/Chrome-Dino-Runner/blob/master/assets/DinoWallpaper.png" width="97" height="97">
  <h2 align="center" style="margin-top: -4px !important;">A Replica of the hidden Dinosaur Game from Chrome Browser Offline mode so you don't have to be offline to play it...😂😂</h2>
  <p align="center">
    <a href="https://github.com/dhhruv/Sudoku-Solver/blob/master/LICENSE">
      <img src="https://img.shields.io/badge/license-MIT-informational">
    </a>
    <a href="https://www.python.org/">
    	<img src="https://img.shields.io/badge/python-v3.8-informational">
    </a>
  </p>
</p>
<p align="center">
	<img src="http://ForTheBadge.com/images/badges/made-with-python.svg">
</p>
<p align="center">   
	<a href="https://dev.to/dhhruv/chrome-dino-game-using-python-2595">
    	<img src="https://img.shields.io/badge/dev.to-0A0A0A?style=for-the-badge&logo=dev.to&logoColor=white">
    </a>
</p>


### Introduction:

-	The Dinosaur Game, also known as the **T-Rex Game, Steve the Jumping Dinosaur, or Dino Runner** and initially codenamed Project Bolan, is a built-in browser game in the **Google Chrome Web Browser**. The game was created by **Sebastien Gabriel in 2014**, and can be accessed by hitting the space bar when in offline mode on Google Chrome.

### About:

-	The following represents a recreated version of the famous Dinosaur Game from Chrome Browser Offline mode implemented using **Python and PyGame**. The project file contains **Image Files** and a python script **(chromedino.py)**.
-	A simple and easy-to-use GUI is provided for better gameplay. The gameplay design is so simple that the user won’t find it difficult to use and understand. Different images are used in the development of this simple game project, the gaming environment is just like the original Chrome Dino Run game. For demo of the project, have a look at the GIF below.

<p align="center">
  <img src="https://github.com/LadinaM/Chrome-Dino-Runner/blob/master/assets/Other/Chrome%20Dino.gif">
</p>

### Project Structure:

```
chrome-dino-runner/
├── assets/                # Game assets (images, sprites)
│   ├── Bird/              # Bird obstacle sprites
│   ├── Cactus/            # Cactus obstacle sprites
│   ├── Dino/              # Dinosaur character sprites
│   └── Other/             # Other game assets
├── figures/               # Game character modules
│   ├── __init__.py        # Package initialization
│   ├── dinosaur.py        # Dinosaur character class
│   ├── cloud.py           # Cloud background class
│   ├── obstacles.py       # Obstacle classes (Cactus, Bird)
│   └── configurations.py  # Game assets and configuration
├── game/                  # Shared game settings and utilities
│   ├── __init__.py        # Package initialization
│   └── settings.py        # Shared game settings, state, and rendering
├── utilities/             # Helpers
    ├── helpers.py         # Utility functions
    ├── constants.py       # Helper classes
    └── observations.py    # Observation normalizer and seed setter
├── chrome_dino.py         # Main game using shared settings
├── chrome_dino_env.py     # RL environment
├── train_dino_agent.py    # Training script
├── test_env.py            # Environment testing script
├── dino_ppo.pt            # 
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

### Installation:

-	First of all, **clone the repository** using:
```
git clone https://github.com/LadinaM/chrome-dino-runner.git
``` 
**OR**
Download the Zip and extract it's contents.

-	Then download the dependencies in your Command Prompt OR Terminal using:
```
pip install -r requirements.txt
```

### Usage:

-	After installation, just run the [`chromedino.py`](https://github.com/LadinaM/chrome-dino-runner/blob/master/chromedino.py) using
```
python chrome_dino.py
```

### Code Refactoring:

The project has been refactored to reduce code duplication between the main game and RL environment:

- **`game/settings.py`** - Contains shared game settings, state management, and rendering functions
- **`chrome_dino.py`** - Main game
- **`chrome_dino_env.py`** - RL environment

### Input:

| Keys              | Actions                                                       |
|-------------------|---------------------------------------------------------------|
|  `Any Key`        |    Press any key to start the Game.                           | 
|   **&#8593;**     |    Press `Up Arrow` to jump and avoid cacti.                  |
|   **&#8595;**     |    Press `Down Arrow` to duck and avoid pterodactyls.         |
|   `p`             |    Pause the game      |
|   `ESC`           |    Pause the game (alternative to 'p')      |
|   `u`             |    Unpause the game     |

### Reinforcement Learning Features:

This project includes a reinforcement learning environment for training AI agents to play the Chrome Dino game automatically.

#### Files:
- `chrome_dino_env.py` - Gymnasium environment for RL training
- `train_dino_agent.py` - Training script using Stable-Baselines3 PPO
- `test_env.py` - Environment testing script

#### Training the AI Agent:

1. **Install additional dependencies:**
```bash
pip install -r requirements.txt
```

2. **Train the agent:**
```bash
python train_dino_agent.py
```

The training script will:
- Train a PPO agent for 100'000 timesteps
- Save checkpoints every 5'000 timesteps
- Save the best model based on evaluation performance
- Test the trained model on 5 episodes

#### Monitoring Training with TensorBoard:

TensorBoard provides real-time visualization of training metrics.

1. **Start TensorBoard:**
```bash
tensorboard --logdir=./tensorboard_logs --port=6006
```

2. **View in browser:**
Open http://localhost:6006 in your web browser

3. **Available metrics:**
- **Scalars** - Training curves for:
  - Episode reward
  - Episode length
  - Loss values (policy loss, value loss, entropy loss)
  - Learning rate
  - Explained variance
- **Actions** - Action distribution over time
- **Environment Info** - Game speed, score, obstacle count

#### Action Space:
The RL environment supports 3 actions:
- **Action 0**: Do nothing
- **Action 1**: Jump (avoid cacti)
- **Action 2**: Duck (avoid birds)

#### Observation Space:
The agent receives a 6-dimensional state vector:
- Dino Y position
- Dino velocity
- Nearest obstacle X position
- Nearest obstacle Y position
- Nearest obstacle type (0=cactus, 1=large cactus, 2=bird)
- Game speed

#### Model Files:
- `dino_final_model` - Final trained model
- `best_model/` - Best model based on evaluation
- `checkpoints/` - Training checkpoints
- `tensorboard_logs/` - Training logs for TensorBoard

#### Testing the Trained Model:
```bash
python test_env.py
```

This will load the trained model and run it in the game environment for evaluation.


### References:
-	http://www.pygame.org/docs
-	https://en.wikipedia.org/wiki/Dinosaur_Game
-	Various articles and videos.
