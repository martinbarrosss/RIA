# train_ppo.py

import os
from robobo_env import RoboboFollowerEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

# --- CONFIGURACIÓN DEL ENTRENAMIENTO ---
LOG_DIR = "./robobo_logs/"
MODEL_DIR = "./robobo_models/"
TOTAL_TIMESTEPS = 10000 # Número de pasos de tiempo para entrenar [cite: 59]
SAVE_FREQ = 1000 # Guardar un checkpoint cada 10,000 pasos

# Crear directorios si no existen
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# Crear el entorno (Vectorizado)
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.monitor import Monitor

def make_env():
    def _init():
        env = RoboboFollowerEnv()
        return Monitor(env)
    return _init

env = DummyVecEnv([make_env()])
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_reward=10.0)


# --- Definición de la Política y el Algoritmo [cite: 44] ---
# Algoritmo: PPO (sugerido) 
# Política: MlpPolicy (Red Neuronal Multicapa)
model = PPO(
    "MlpPolicy", 
    env, 
    verbose=1,
    tensorboard_log=LOG_DIR, # Para guardar las métricas de entrenamiento [cite: 64]
    learning_rate=3e-4, 
    n_steps=2048, # Pasos de tiempo antes de la actualización de la política [cite: 59]
)

# Callback para guardar el modelo periódicamente (Checkpoint)
checkpoint_callback = CheckpointCallback(
    save_freq=SAVE_FREQ,
    save_path=MODEL_DIR,
    name_prefix="robobo_ppo_checkpoint",
    save_replay_buffer=False,
    save_vecnormalize=False,
)

print(f"Iniciando entrenamiento con {TOTAL_TIMESTEPS} pasos de tiempo...")

# Entrenar el modelo
# El bucle de la Figura 2 se ejecuta automáticamente aquí [cite: 58]
model.learn(
    total_timesteps=TOTAL_TIMESTEPS, 
    callback=checkpoint_callback
)

# Guardar el modelo final
model.save(f"{MODEL_DIR}/robobo_ppo_final.zip")
print(f"Entrenamiento completado y modelo guardado en {MODEL_DIR}/robobo_ppo_final.zip")

env.close()