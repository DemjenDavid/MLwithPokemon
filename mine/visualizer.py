# Run a single environment for visualization
import time
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pyboy import PyBoy
from stable_baselines3 import PPO

class PyBoyPokemonEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, rom_path, render_mode="rgb_array"):
        super().__init__()
        self.p_p_action = None
        self.p_action = None
        self.pyboy = PyBoy(
            rom_path,
            window="null" if render_mode != "human" else "SDL2",
            debug=False
        )
        with open("roms/state_file.state", "rb") as f:
            self.pyboy.load_state(f)
        self.pyboy.set_emulation_speed(1)
        # Action space: Up, Down, Left, Right, A, B, Start, Select
        self.buttons = [
            "up", "down", "left", "right",
            "a", "b", "start", "select"
        ]
        self.action_space = spaces.Discrete(len(self.buttons))

        # Observation: raw Game Boy screen (160×144 RGB
        frame = self.pyboy.screen.ndarray 
        H, W, C = 144, 160, 3  # after dropping alpha, C=3
        self.observation_space = spaces.Box(low=0, high=255, shape=(H, W, C), dtype=np.uint8)
        self.render_mode = render_mode

    def step(self, action):
        # Press selected button for a few frames
        screen = self.pyboy.game_area()

        button = self.buttons[action]
        self.pyboy.button_press(button)
        self.pyboy.tick(4)
        self.pyboy.button_release(button)
        self.pyboy.tick(16)
        # Grab screen
        print("Frame shape:", self.pyboy.game_area())
        print(f"Action taken: {button}")
        print(f"Difference in screen sum: {np.abs(np.sum(screen) - np.sum(self.pyboy.game_area()))} ")
        # TODO: Define a meaningful reward function
        # if some change on the screen:
        
        if np.abs(np.sum(screen) - np.sum(self.pyboy.game_area())) > 1000:
            reward = 1.0  # reward for screen change
        elif action == self.p_p_action and action == self.p_action:
                reward = -1.0  # small reward for repeating the same action
        else:
            reward = -0.1  # default reward

        terminated = False
        truncated = False
        self.p_p_action = self.p_action
        self.p_action = action
        return self._get_frame(), reward, terminated, truncated, {}
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Reset emulator
        # self.pyboy.stop()
        obs = self._get_frame()
        return obs, {}

    def render(self):
        if self.render_mode == "human":
            pass  # PyBoy manages its own SDL window

    def _get_frame(self):
        frame = self.pyboy.screen.ndarray[..., :3]  # keep only RGB
        return frame.astype(np.uint8)


model = PPO.load("./ppo_pokemon_blue.zip") 
visual_env = PyBoyPokemonEnv("roms/Pokemon_Blue.gb", render_mode="human")
obs, info = visual_env.reset()
print(f"Starting....")
for _ in range(100):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = visual_env.step(action)
    visual_env.render()  # SDL2 window updates
    if terminated or truncated:
        obs, info = visual_env.reset()
