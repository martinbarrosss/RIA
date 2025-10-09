# train_ppo.py

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from robobo_env import RoboboFollowerEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# --- CONFIGURACIÓN ---
LOG_DIR = "./robobo_logs/"
MODEL_DIR = "./robobo_models/"
TOTAL_TIMESTEPS = 10000
N_STEPS = 50

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# 📊 Callback personalizado para registrar métricas y posiciones
class TrainingLoggerCallback(BaseCallback):

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_means = []
        self.episode_lengths = []
        self.positions = []

    def _on_step(self) -> bool:
        # Guardar posición actual del robot
        try:
            robobo_state = self.training_env.envs[0].env.sim.getRobotLocation(0)
            if robobo_state and "position" in robobo_state:
                x = robobo_state["position"]["x"]
                z = robobo_state["position"]["z"]
                self.positions.append((x, z))
        except Exception:
            pass
        return True

    def _on_rollout_end(self):
        # Cada vez que termina un rollout, guardamos métricas de episodios completados
        info_buffer = self.locals.get("infos", [])
        for info in info_buffer:
            if "episode" in info:
                total_reward = info["episode"]["r"]
                length = info["episode"]["l"]
                mean_reward = total_reward / length if length > 0 else 0
                self.episode_rewards.append(total_reward)
                self.episode_lengths.append(length)
                self.episode_means.append(mean_reward)

    def _on_training_end(self):
        print("\n📊 Entrenamiento finalizado. Generando métricas...")

        # Crear DataFrame con resultados
        df = pd.DataFrame({
            "Episodio": np.arange(1, len(self.episode_rewards) + 1),
            "Recompensa Total": self.episode_rewards,
            "Recompensa Media": self.episode_means,
            "Longitud Episodio": self.episode_lengths
        })
        df.to_csv("training_metrics.csv", index=False)
        print("✅ Métricas guardadas en training_metrics.csv")

        sns.set(style="whitegrid", font_scale=1.2)

        # 📈 Recompensas
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x="Episodio", y="Recompensa Total", marker="o", label="Total")
        sns.lineplot(data=df, x="Episodio", y="Recompensa Media", marker="s", label="Media")
        plt.title("📈 Recompensas durante el entrenamiento")
        plt.xlabel("Episodio")
        plt.ylabel("Recompensa")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("training_rewards.png", dpi=300)
        plt.show()

        # 📉 Longitud del episodio
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x="Episodio", y="Longitud Episodio", marker="o", color="darkorange")
        plt.title("📉 Longitud de los episodios durante el entrenamiento")
        plt.xlabel("Episodio")
        plt.ylabel("Número de pasos")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("training_episode_length.png", dpi=300)
        plt.show()

        # 📍 Trayectoria 2D
        if self.positions:
            xs, zs = zip(*self.positions)
            plt.figure(figsize=(8, 8))
            plt.scatter(xs, zs, s=10, c=np.linspace(0, 1, len(xs)), cmap="viridis", alpha=0.7)
            plt.title("📍 Posiciones del Robobo durante el entrenamiento")
            plt.xlabel("Posición X")
            plt.ylabel("Posición Z")
            plt.grid(True)
            plt.savefig("training_positions.png", dpi=300)
            plt.show()
            print("✅ Trayectoria guardada en training_positions.png")
        else:
            print("⚠️ No se registraron posiciones del robot.")

# --- CREACIÓN DEL ENTORNO ---
def make_env():

    def _init():
        env = RoboboFollowerEnv(discrete_actions=False)
        return Monitor(env)
    return _init

env = DummyVecEnv([make_env()])

# --- ENTRENAMIENTO PPO ---
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log=LOG_DIR,
    learning_rate=3e-2,
    n_steps=N_STEPS,
)

print(f"🚀 Iniciando entrenamiento con {TOTAL_TIMESTEPS} pasos...")

callback = TrainingLoggerCallback(verbose=1)

model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=callback
)

# Guardar modelo final
model.save(f"{MODEL_DIR}/robobo_ppo_final.zip")
print(f"✅ Modelo guardado en {MODEL_DIR}/robobo_ppo_final.zip")

env.close()
