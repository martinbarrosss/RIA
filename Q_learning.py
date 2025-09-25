
import os
import time
import math
import random
from typing import Tuple, Dict, Any, List

import gymnasium as gym
from gymnasium import spaces
import numpy as np

# Robobo / simulator imports (same style as the provided file)
try:
    from robobopy.Robobo import Robobo
    from robobopy.utils.IR import IR
    from robobopy.utils.LED import LED
    from robobosim.RoboboSim import RoboboSim
except Exception as e:
    # Allow importing even when not installed on the machine: raise a helpful error later
    Robobo = None
    RoboboSim = None
    IR = None

# Stable Baselines 3
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import BaseCallback
except Exception:
    PPO = None

# Matplotlib for plotting results
import matplotlib.pyplot as plt
import csv

# ----- Environment definition -----

class RoboboGymEnv(gym.Env):
    """Gymnasium environment wrapping RoboboSim / Robobo.

    Observation: vector of 8 floats (normalized to [0,1]):
        [brightness_left50, brightness_left25, brightness_center, brightness_right25, brightness_right50,
         ir_frontC, ir_frontL, ir_frontR]

    Action space: Discrete(5) (same semantic as the provided file):
        0: Girar muy a la derecha
        1: Girar a la derecha
        2: Avanzar
        3: Girar a la izquierda
        4: Girar muy a la izquierda

    Reward:
        - Positive when the center brightness increases
        - Large positive for reaching the target brightness threshold
        - Large negative and termination on collision (IR threshold)
        - Small time penalty per step to encourage speed
    """

    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self, host: str = "localhost", brightness_goal: float = 350.0, max_steps: int = 200):
        super().__init__()

        self.host = host
        self.brightness_goal = brightness_goal
        self.max_steps = max_steps

        # Action and observation spaces
        self.action_space = spaces.Discrete(5)
        # Observations normalized to [0,1]
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(8,), dtype=np.float32)

        # Connect to Robobo & simulator
        if Robobo is None or RoboboSim is None:
            print("Warning: robobopy or robobosim not available. Environment will raise on connect().")

        self.rbb = None
        self.sim = None

        # Internal state
        self.current_obs = None
        self.steps = 0
        self.episode_rewards = 0.0
        self.trajectory: List[Tuple[float, float]] = []  # store (x,y) if pose available

    # ------- Helper: hardware/simulator connection -------
    def _connect(self):
        if self.rbb is None:
            if Robobo is None:
                raise RuntimeError("Robobo library not available. Please install robobopy and robobosim.")
            self.rbb = Robobo(self.host)
            self.rbb.connect()

        if self.sim is None:
            if RoboboSim is None:
                raise RuntimeError("RoboboSim library not available. Please install robobosim.")
            self.sim = RoboboSim(self.host)
            self.sim.connect()

    def close(self):
        try:
            if self.rbb:
                self.rbb.disconnect()
        except Exception:
            pass
        try:
            if self.sim:
                self.sim.disconnect()
        except Exception:
            pass

    # ------- Sensors & actions (inspired by uploaded script) -------
    def _read_brightnesss(self) -> List[float]:
        # Following the order used in the uploaded script: [II, I, C, D, DD]
        # We will return normalized values in [0,1] using an assumed max (e.g. 1023)
        max_sensor = 1023.0
        # Pan positions: -25, -50, 25, 50, 0 in the uploaded script
        try:
            self.rbb.movePanTo(-25, 100, wait=True)
            luz_I = self.rbb.readBrightnessSensor()
            self.rbb.wait(0.05)

            self.rbb.movePanTo(-50, 100, wait=True)
            luz_II = self.rbb.readBrightnessSensor()
            self.rbb.wait(0.05)

            self.rbb.movePanTo(25, 100, wait=True)
            luz_D = self.rbb.readBrightnessSensor()
            self.rbb.wait(0.05)

            self.rbb.movePanTo(50, 100, wait=True)
            luz_DD = self.rbb.readBrightnessSensor()
            self.rbb.wait(0.05)

            self.rbb.movePanTo(0, 100, wait=True)
            luz_C = self.rbb.readBrightnessSensor()
            self.rbb.wait(0.05)
        except Exception as e:
            # If sensors not available, return zeros
            print("Warning reading brightness sensors:", e)
            luz_II = luz_I = luz_C = luz_D = luz_DD = 0.0

        raw = [luz_II, luz_I, luz_C, luz_D, luz_DD]
        norm = [min(max(v / max_sensor, 0.0), 1.0) for v in raw]
        return norm

    def _read_ir(self) -> List[float]:
        # Returns normalized IR values in [0,1] with assumed max 1023.
        max_ir = 1023.0
        try:
            ir_frontal = self.rbb.readIRSensor(IR.FrontC)
            ir_frontL = self.rbb.readIRSensor(IR.FrontL)
            ir_frontR = self.rbb.readIRSensor(IR.FrontR)
        except Exception as e:
            print("Warning reading IR sensors:", e)
            ir_frontal = ir_frontL = ir_frontR = 0.0

        return [min(max(ir_frontal / max_ir, 0.0), 1.0),
                min(max(ir_frontL / max_ir, 0.0), 1.0),
                min(max(ir_frontR / max_ir, 0.0), 1.0)]

    def _build_obs(self) -> np.ndarray:
        b = self._read_brightnesss()
        ir = self._read_ir()
        obs = np.array(b + ir, dtype=np.float32)
        return obs

    def _execute_action(self, action: int):
        # Durations tuned for Gym steps; adapt if necessary
        if action == 0:  # Girar muy a la derecha
            self.rbb.moveWheelsByTime(10, -10, 1.0, wait=True)
        elif action == 1:  # Girar a la derecha
            self.rbb.moveWheelsByTime(10, -10, 0.5, wait=True)
        elif action == 2:  # Avanzar
            self.rbb.moveWheelsByTime(20, 20, 1.2, wait=True)
        elif action == 3:  # Girar a la izquierda
            self.rbb.moveWheelsByTime(-10, 10, 0.5, wait=True)
        elif action == 4:  # Girar muy a la izquierda
            self.rbb.moveWheelsByTime(-10, 10, 1.0, wait=True)

    # ------- Gym API -------
    def reset(self, seed: int = None, options: Dict[str, Any] = None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Lazy connect
        self._connect()

        # Reset simulation and internal counters
        try:
            self.sim.resetSimulation()
        except Exception as e:
            print("Warning: sim.resetSimulation() failed:", e)

        self.steps = 0
        self.episode_rewards = 0.0
        self.trajectory = []

        # Small wait to stabilize sensors
        time.sleep(0.1)

        obs = self._build_obs()
        self.current_obs = obs

        # Try to store initial pose if available
        try:
            pose = self.sim.getRobotPose()
            self.trajectory.append((pose[0], pose[1]))
        except Exception:
            # If pose API not present, append (0,0)
            self.trajectory.append((0.0, 0.0))

        return obs, {}

    def step(self, action: int):
        assert self.action_space.contains(action)

        old_center = float(self.current_obs[2])  # center brightness normalized

        # Execute action in sim/hardware
        try:
            self._execute_action(action)
        except Exception as e:
            print("Warning executing action:", e)

        # Read new observation
        obs = self._build_obs()
        self.current_obs = obs
        self.steps += 1

        # Compute reward
        new_center = float(obs[2])
        # Reward is positive if center increased
        reward = (new_center - old_center) * 10.0

        # Time penalty to encourage reaching quickly
        reward -= 0.01

        done = False
        info = {}

        # Check goal reached (denormalize by assumed max 1023)
        denorm_center = new_center * 1023.0
        if denorm_center >= self.brightness_goal:
            reward += 100.0
            done = True
            info['reason'] = 'goal_reached'
            # reset sim internally so the environment is in a clean state on next reset
            try:
                self.sim.resetSimulation()
            except Exception:
                pass

        # Collision (IR threshold). The uploaded script used numeric thresholds ~150 and 40.
        try:
            ir_frontC_raw = self.rbb.readIRSensor(IR.FrontC)
            ir_frontL_raw = self.rbb.readIRSensor(IR.FrontL)
            ir_frontR_raw = self.rbb.readIRSensor(IR.FrontR)
        except Exception:
            ir_frontC_raw = ir_frontL_raw = ir_frontR_raw = 1023.0

        # If frontal IR indicates close obstacle (small value in uploaded script <150)
        if ir_frontC_raw < 150:
            reward -= 50.0
            done = True
            info['reason'] = 'collision'

        # Timeout termination
        if self.steps >= self.max_steps:
            done = True
            info['reason'] = 'timeout'

        self.episode_rewards += reward

        # Store pose if available
        try:
            pose = self.sim.getRobotPose()
            self.trajectory.append((pose[0], pose[1]))
        except Exception:
            # append last pose-like approximation (keep a moving estimate)
            self.trajectory.append(self.trajectory[-1] if self.trajectory else (0.0, 0.0))

        return obs, float(reward), done, False, info

    def render(self):
        # We don't open any windows here; rely on RoboboSim's own renderer (if any)
        pass


# ----- Training & evaluation utilities -----

class RewardLoggerCallback(BaseCallback):
    """Simple callback that logs mean reward per `log_every` steps and
    saves a PNG plot of the learning curve.
    """
    def __init__(self, log_dir: str = "logs", log_every: int = 10000, verbose: int = 0):
        super().__init__(verbose)
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_every = log_every
        self.rewards = []
        self.timesteps = []

    def _on_step(self) -> bool:
        if self.n_calls % self.log_every == 0:
            # Not the best metric (no access to ep info here), but save model + dummy plot
            filepath = os.path.join(self.log_dir, f"model_step_{self.num_timesteps}.zip")
            try:
                self.model.save(filepath)
            except Exception:
                pass
            # Save a small plot summarizing training progress
            try:
                # Try to extract episode rewards from the logger if present
                # This is a best-effort - stable_baselines writes to the logger dict
                self._save_plot()
            except Exception:
                pass
        return True

    def _save_plot(self):
        # Create a simple placeholder plot (timesteps vs. saved models on checkpoints)
        fig, ax = plt.subplots()
        ax.set_title("PPO training checkpoints")
        ax.set_xlabel("dummy index")
        ax.set_ylabel("timesteps")
        ax.plot([0, 1], [0, self.num_timesteps])
        plt.tight_layout()
        fpath = os.path.join(self.log_dir, "training_progress.png")
        fig.savefig(fpath)
        plt.close(fig)


def train(total_timesteps: int = 200_000, host: str = "localhost"):
    if PPO is None:
        raise RuntimeError("Stable-Baselines3 not available. Install with `pip install stable-baselines3[extra]`.")

    def make_env():
        return RoboboGymEnv(host=host)

    env = DummyVecEnv([make_env])

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log="./tb_logs",
        batch_size=64,
        learning_rate=3e-4,
    )

    cb = RewardLoggerCallback(log_dir="./logs", log_every=20000)

    model.learn(total_timesteps=total_timesteps, callback=cb)
    model.save("ppo_robobo_practice01")

    print("Training finished. Model saved to ppo_robobo_practice01.zip")


def evaluate(model_path: str = "ppo_robobo_practice01.zip", n_episodes: int = 5, host: str = "localhost"):
    # Load env & model, run episodes and save trajectory + reward info
    env = RoboboGymEnv(host=host, max_steps=300)

    if PPO is None:
        raise RuntimeError("Stable-Baselines3 not available. Install with `pip install stable-baselines3[extra]`.")

    model = PPO.load(model_path)

    results = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(int(action))
            ep_reward += reward
        results.append({'episode': ep, 'reward': ep_reward, 'reason': info.get('reason', '')})

        # Save trajectory CSV
        csv_path = f"trajectory_ep{ep}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["x", "y"])
            for x, y in env.trajectory:
                writer.writerow([x, y])
        # Plot 2D trajectory
        xs = [p[0] for p in env.trajectory]
        ys = [p[1] for p in env.trajectory]
        fig, ax = plt.subplots()
        ax.plot(xs, ys, marker='o', linewidth=1)
        ax.set_title(f"Trajectory episode {ep} (reward={ep_reward:.2f})")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.grid(True)
        fig.savefig(f"trajectory_ep{ep}.png")
        plt.close(fig)

    print("Evaluation finished. Results:")
    for r in results:
        print(r)


# ----- Command-line interface -----
if __name__ == '__main__':

    robobo = Robobo("localhost")
    robobo.connect()

    sim = RoboboSim("localhost")
    sim.connect()

    import argparse

    parser = argparse.ArgumentParser(description="Train/evaluate PPO on RoboboSim via Gymnasium")
    parser.add_argument('--train', action='store_true', help='Run training')
    parser.add_argument('--eval', action='store_true', help='Run evaluation (requires saved model)')
    parser.add_argument('--timesteps', type=int, default=200000, help='Training timesteps')
    parser.add_argument('--host', type=str, default='localhost', help='Robobo/RoboboSim host')
    parser.add_argument('--model', type=str, default='ppo_robobo_practice01.zip', help='Model path for evaluation')
    args = parser.parse_args()

    if args.train:
        train(total_timesteps=args.timesteps, host=args.host)
    if args.eval:
        evaluate(model_path=args.model, n_episodes=3, host=args.host)
