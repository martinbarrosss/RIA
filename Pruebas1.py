"""
Entrenamiento PPO para RoboboSim + Gymnasium
Archivo único que contiene:
 - Clase envRoboboGym: entorno compatible con Gymnasium (observaciones y acciones continuas)
 - Script de entrenamiento usando Stable-Baselines3 (PPO)
 - Callback de evaluación y guardado de métricas + gráficos (PNG)

INSTRUCCIONES RÁPIDAS:
 - Instalar dependencias: pip install gymnasium stable-baselines3 robobopy robobosim matplotlib pandas
 - Ejecutar: python envRobobo_train_PPO.py

ADVERTENCIAS:
 - El código usa métodos de la API de robobopy/robobosim similares a los ejemplos que proporcionaste. Si alguno de los nombres/firmas difiere en tu instalación, adapta las llamadas (comentarios marcados con TODO).
 - He incluido protecciones por si el blob no está detectado o algunos sensores no existen. Ajusta umbrales y normalizaciones según tu simulador.
"""

import os
import time
import math
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Robobo imports (usar los mismos que tú usas)
try:
    from robobopy.Robobo import Robobo
    from robobosim.RoboboSim import RoboboSim
    from robobopy.utils.BlobColor import BlobColor
    from robobopy.utils.IR import IR
except Exception as e:
    # Si no estás ejecutando en el entorno con robobo instalado, las imports fallarán.
    # Mantenemos esto para que el archivo sea editable offline.
    Robobo = None
    RoboboSim = None
    BlobColor = None
    IR = None
    print("Aviso: No se pudieron importar módulos Robobo. Asegúrate de ejecutar esto donde esté instalado.")

# Stable Baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

# Visualización
import matplotlib.pyplot as plt
import pandas as pd


class envRoboboGym(gym.Env):
    """Entorno Gymnasium para Robobo (simulación)
    Observación (vector continuo):
        [posx_norm, posy_norm, centering_error, visible_flag, ir_frontC_norm]
    Acción (vector continuo 2):
        [left_wheel_speed, right_wheel_speed] en rango [-1, 1] que se escalan dentro del env a rpm/vel.

    Recompensa:
        - Fuerte recompensa por acercarse (posy_norm alto)
        - Penalización por estar fuera del centro (abs centering error)
        - Penalización por perder el blob
        - Penalización por colisiones detectadas por IR
    """

    metadata = {"render.modes": []}

    def __init__(self, sim_address: str = 'localhost'):
        super().__init__()

        # Conexiones: ahora la conexión se hace de forma perezosa (lazy) en reset()
        # para evitar múltiples ciclos de connect/disconnect al crear varios envs (DummyVecEnv, EvalCallback, etc.)
        self.robobo = None
        self.sim = None
        self.sim_address = sim_address

        # Spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        # posx, posy, centering_error, visible, ir_frontC
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(5,), dtype=np.float32)

        # parámetros
        self.steps = 0
        self.max_steps = 200
        self.wheel_scale = 40.0  # multiplicador para convertir acción [-1,1] a velocidad de ruedas
        self.step_duration = 0.3  # segundos por step

        # seguridad
        self.ir_collision_threshold = 120  # valor aproximado; ajusta según tus sensores

        # estado inicial
        self.last_blob = None

    def _ensure_connected(self):
        """Conecta a Robobo y al simulador si no están conectados. Evita reconexiones innecesarias."""
        if self.robobo is not None and self.sim is not None:
            return
        try:
            if Robobo is not None and self.robobo is None:
                print("Attempting Robobo connection...")
                self.robobo = Robobo(self.sim_address)
                self.robobo.connect()
                print("Robobo connected")
            if RoboboSim is not None and self.sim is None:
                print("Attempting RoboboSim connection...")
                self.sim = RoboboSim(self.sim_address)
                self.sim.connect()
                print("RoboboSim connected")
        except Exception as e:
            print(f"Warning: connection attempt failed: {e}")

    def _read_blob(self):

        """Lee blob rojo desde la cámara. Devuelve diccionario con posx,posy,size o None."""
        if self.robobo is None:
            return None
        try:
            blob = self.robobo.readColorBlob(BlobColor.RED)
            if blob is None:
                return None
            # Algunos entornos devuelven posx en rango 0..100, posy 0..100. Normalizamos.
            posx = getattr(blob, 'posx', getattr(blob, 'x', None))
            posy = getattr(blob, 'posy', getattr(blob, 'y', None))
            size = getattr(blob, 'area', getattr(blob, 'size', 1.0))
            if posx is None or posy is None:
                return None
            return {"posx": float(posx), "posy": float(posy), "size": float(size)}
        except Exception as e:
            print(f"_read_blob error: {e}")
            return None

    def _read_ir_front(self):
        if self.robobo is None or IR is None:
            return None
        try:
            v = self.robobo.readIRSensor(IR.FrontC)
            return float(v)
        except Exception as e:
            print(f"_read_ir_front error: {e}")
            return None

    def _normalize_blob(self, blob):
        """Convierte blob (posx,posy) crudo a normalizado 0..1 en los ejes de imagen.
        Si no hay blob, devuelve valores neutros (0,0,1 para visible flag 0)
        """
        if blob is None:
            return 0.0, 0.0, 1.0  # posx_norm, posy_norm, visible_flag(=0 represented as 1?)
        # Suposición: posx/posy en rango 0..100. Ajusta si tu simulador es distinto.
        posx = np.clip(blob['posx'] / 100.0, 0.0, 1.0)
        posy = np.clip(blob['posy'] / 100.0, 0.0, 1.0)
        return posx, posy, 1.0

    def _build_observation(self, blob_raw, ir_front):
        posx, posy, visible = self._normalize_blob(blob_raw)
        centering = abs(posx - 0.5)  # 0 es centrado, 0.5 es borde
        ir_norm = 0.0
        if ir_front is not None:
            # Normalizar por un rango asumido 0..400 -> 0..1
            ir_norm = float(np.clip(ir_front / 400.0, 0.0, 1.0))
        visible_flag = 1.0 if blob_raw is not None else 0.0
        obs = np.array([posx, posy, centering, visible_flag, ir_norm], dtype=np.float32)
        return obs

    def step(self, action):
        """Ejecuta acción y devuelve tupla gym: obs, reward, terminated, truncated, info"""
        # Clip action
        action = np.clip(action, -1.0, 1.0)
        left_speed = float(action[0]) * self.wheel_scale
        right_speed = float(action[1]) * self.wheel_scale

        # Ejecutar motores
        try:
            if self.robobo is not None:
                # Usamos moveWheelsByTime para ejecutar una micro-acción
                self.robobo.moveWheelsByTime(left_speed, right_speed, self.step_duration)
        except Exception as e:
            print(f"Error moviendo ruedas: {e}")

        # Leer sensores
        blob = self._read_blob()
        ir_front = self._read_ir_front()
        obs = self._build_observation(blob, ir_front)

        # Recompensa base
        reward = 0.0
        done = False
        terminated = False
        truncated = False
        info = {}

        # Si hay blob visible, fomentar acercamiento (posy grande) y centrar en x
        if blob is not None:
            # asumimos posy_norm en 0..1, con valores mayores = más cerca
            _, posy_norm, _ = self._normalize_blob(blob)
            centering = obs[2]
            reward += 2.0 * posy_norm  # incentiva acercarse
            reward += -1.0 * centering  # penaliza estar fuera del centro
        else:
            # penaliza no ver el objetivo
            reward -= 0.5

        # Penalizar si IR detecta obstáculo cercano
        collision = False
        if ir_front is not None and ir_front < self.ir_collision_threshold:
            # cuanto más pequeño, más cerca la colisión; aquí penalizamos
            reward -= 2.0
            collision = True

        # Penalizar acciones muy bruscas (promueve suavidad)
        action_penalty = 0.01 * (abs(action[0]) + abs(action[1]))
        reward -= action_penalty

        # Comprobar condiciones de finalización
        # 1) Si estamos lo suficientemente cerca (posy_norm alto)
        close_threshold = 0.88
        if blob is not None:
            _, posy_norm, _ = self._normalize_blob(blob)
            if posy_norm >= close_threshold:
                reward += 10.0
                terminated = True
                info['reason'] = 'reached_object'

        # 2) Si colisionamos
        if collision:
            reward -= 5.0
            terminated = True
            info['reason'] = 'collision'

        # 3) truncado por pasos max
        self.steps += 1
        if self.steps >= self.max_steps:
            truncated = True
            info['reason'] = info.get('reason', 'max_steps')

        return obs, float(reward), terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        # gymnasium API
        super().reset(seed=seed)

        # Asegurar conexión (lazy connect) - evita ciclos constantes de connect/disconnect
        self._ensure_connected()

        # Reset simulador si disponible
        try:
            if self.sim is not None:
                self.sim.resetSimulation()
            # No desconectamos/reconectamos aquí: eso provoca que múltiples wrappers
            # (Monitor, DummyVecEnv, EvalCallback) abran/cerren conexiones repetidamente.
            if self.robobo is not None:
                # mover tilt/pan a posición neutra si existe la API
                try:
                    self.robobo.moveTiltTo(100, 30, wait=True)
                except Exception:
                    pass
        except Exception as e:
            print(f"Reset error: {e}")

        self.steps = 0
        self.last_blob = None

        # Leer observación inicial
        blob = self._read_blob()
        ir_front = self._read_ir_front()
        obs = self._build_observation(blob, ir_front)
        return obs, {}

    def close(self):
        # Desconectar solo cuando el entorno realmente se cierra
        try:
            if self.robobo is not None:
                try:
                    self.robobo.disconnect()
                    print("Robobo disconnected")
                except Exception as e:
                    print(f"Error disconnecting Robobo: {e}")
                self.robobo = None
            if self.sim is not None:
                try:
                    self.sim.disconnect()
                    print("RoboboSim disconnected")
                except Exception as e:
                    print(f"Error disconnecting RoboboSim: {e}")
                self.sim = None
        except Exception as e:
            print(f"Close error: {e}")

        try:
            if self.robobo is not None:
                self.robobo.disconnect()
            if self.sim is not None:
                self.sim.disconnect()
        except Exception as e:
            print(f"Close error: {e}")


# ------------------ Entrenamiento PPO ------------------

def make_env_fn():
    def _init():
        env = envRoboboGym('localhost')
        # Monitor wrapper guarda estadísticas (monitor.csv)
        return Monitor(env)
    return _init


def plot_monitor_csv(monitor_file, out_png):
    """Lee monitor.csv (formato Monitor de SB3) y genera un PNG con la media móvil de reward."""
    try:
        data = pd.read_csv(monitor_file, skiprows=1)
    except Exception as e:
        print(f"No se pudo leer {monitor_file}: {e}")
        return

    if 'r' not in data.columns:
        print("El monitor.csv no tiene la columna 'r' (reward)")
        return

    rewards = data['r'].values
    episodes = np.arange(len(rewards))
    window = max(1, int(len(rewards) * 0.05))
    rolling = pd.Series(rewards).rolling(window=window, min_periods=1).mean()

    plt.figure(figsize=(8,4))
    plt.plot(episodes, rewards, alpha=0.3, label='ep_reward')
    plt.plot(episodes, rolling, label='rolling_mean')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training rewards')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    print(f"Guardado gráfico de reward en {out_png}")


if __name__ == '__main__':
    # Parámetros de entrenamiento
    total_timesteps = int(2e5)
    eval_freq = 5000
    n_eval_episodes = 5
    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)

    # Crear entornos
    train_env = DummyVecEnv([make_env_fn()])
    eval_env = DummyVecEnv([make_env_fn()])

    # Stop callback (opcional: parar si alcanza umbral de reward)
    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=50.0, verbose=1)
    eval_callback = EvalCallback(eval_env, best_model_save_path=model_dir,
                                 log_path=model_dir, eval_freq=eval_freq,
                                 n_eval_episodes=n_eval_episodes, callback_after_eval=stop_callback,
                                 verbose=1)

    # Crear modelo PPO
    model = PPO('MlpPolicy', train_env, verbose=1, tensorboard_log='./tensorboard_logs')

    print("Comenzando entrenamiento PPO")
    start_time = time.time()
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    elapsed = time.time() - start_time
    print(f"Entrenamiento finalizado en {elapsed/60:.2f} minutos")

    # Guardar modelo final
    model_path = os.path.join(model_dir, 'ppo_robobo_final.zip')
    model.save(model_path)
    print(f"Modelo guardado en {model_path}")

    # Intentar graficar el monitor.csv generado en model_dir
    monitor_csv = os.path.join(model_dir, 'monitor.csv')
    # Si EvalCallback guardó un log_path, puede haber archivos con nombre monitor.csv. Buscamos el más reciente.
    candidate = None
    for root, dirs, files in os.walk(model_dir):
        for f in files:
            if f.startswith('monitor') and f.endswith('.csv'):
                candidate = os.path.join(root, f)
    if candidate is not None:
        plot_monitor_csv(candidate, os.path.join(model_dir, 'training_rewards.png'))
    else:
        print("No se encontró monitor.csv en el directorio de modelos. Si usas Monitor, revisa dónde se guardó.")

    # Cerrar entornos
    train_env.close()
    eval_env.close()

    print("Script terminado. Puedes cargar el modelo con: PPO.load('models/ppo_robobo_final.zip')")
