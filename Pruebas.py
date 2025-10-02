import argparse
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Robobo imports (asegúrate de tener estas librerías instaladas)
from robobopy.Robobo import Robobo
from robobosim.RoboboSim import RoboboSim
from robobopy.utils.BlobColor import BlobColor
from robobopy.utils.IR import IR

# Stable Baselines imports
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

# Utilidades para logging y plotting
import matplotlib.pyplot as plt
import os


class RoboboGymEnv(gym.Env):
    """Entorno Gym para Robobo en RoboboSim.

    Observación: Box(2) --> [posx_norm, dist_norm]
      - posx_norm: posx del blob mapeada a [-1, 1] (izquierda -> -1, centro 0, derecha -> +1)
      - dist_norm: estimación de distancia normalizada en [0,1] (0 cerca, 1 lejos)

    Acción: Box(2) --> [wheel_left_speed, wheel_right_speed] en rango [-50, 50]
    """

    metadata = {"render.modes": []}

    def __init__(self,
                 host: str = "localhost",
                 max_steps: int = 200,
                 wheel_duration: float = 0.4,
                 max_wheel_speed: float = 50.0,
                 reach_dist_threshold: float = 0.8):
        super().__init__()

        # PRIMERO: definir los espacios (siempre deben existir aunque falle la conexión)
        self.max_wheel_speed = max_wheel_speed
        self.action_space = spaces.Box(low=np.array([-self.max_wheel_speed, -self.max_wheel_speed], dtype=np.float32),
                                    high=np.array([self.max_wheel_speed, self.max_wheel_speed], dtype=np.float32),
                                    dtype=np.float32)

        obs_low = np.array([-1.0, 0.0], dtype=np.float32)
        obs_high = np.array([1.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # Parámetros y estado interno
        self.host = host
        self.max_steps = max_steps
        self.wheel_duration = wheel_duration
        self.reach_dist_threshold = reach_dist_threshold

        # Conexión con Robobo y simulador (intentos no fatales)
        self.robobo = Robobo(self.host)
        self.sim = RoboboSim(self.host)
        self._connect()

        # Estado interno adicional
        self.steps = 0
        self.prev_dist = 1.0
        self.latest_obs = np.array([0.0, 1.0], dtype=np.float32)
        self.trajectory = []

        # Para evaluar y guardar trayectoria
        self.trajectory = []  # lista de (x,y) si se puede obtener del simulador

    def _connect(self):
        try:
            self.robobo.connect()
        except Exception as e:
            print("Aviso: no se pudo conectar con Robobo: ", e)
        try:
            self.sim.connect()
        except Exception as e:
            print("Aviso: no se pudo conectar con RoboboSim: ", e)

    def _get_blob_features(self, blob):
        """Extrae posx y PROXIMIDAD (prox_norm) a partir del blob.
        prox_norm: 0.0 = muy lejos, 1.0 = muy cerca.
        """
        if blob is None:
            return None

        # posx: valor absoluto del eje X en pantalla (ej: 0..100). Se normaliza a [-1,1]
        posx = getattr(blob, 'posx', None)
        if posx is None:
            return None

        # Intentamos sacar un indicador de tamaño/Área del blob
        area = None
        for attr in ('area', 'size', 'width', 'radius', 'height'):
            if hasattr(blob, attr):
                try:
                    area = float(getattr(blob, attr))
                    break
                except Exception:
                    area = None

        # Si tenemos area, la usamos directamente como proximidad (area grande -> cerca)
        if area is not None:
            max_area_estimate = 20000.0  # AJUSTA según tu simulador
            area_clipped = max(0.0, min(max_area_estimate, area))
            prox_norm = float(np.clip(area_clipped / max_area_estimate, 0.0, 1.0))
        else:
            # Fallback: usar sensor IR frontal. Debes saber si tu IR devuelve mayor = más cerca.
            # Aquí asumimos que EN TU SIMULADOR IR ALTA = MÁS CERCA (ajusta si es al revés).
            try:
                ir = self.robobo.readIRSensor(IR.FrontC)
                if ir is None:
                    prox_norm = 0.0
                else:
                    max_ir = 700.0  # AJUSTA según tu hardware/simulador
                    ir_val = float(ir)
                    prox_norm = float(np.clip(ir_val / max_ir, 0.0, 1.0))
            except Exception:
                prox_norm = 0.0

        # Normalizar posx: asumimos posx en [0,100] -> mapear a [-1,1]
        try:
            posx_f = float(posx)
            posx_norm = (posx_f - 50.0) / 50.0
            posx_norm = float(np.clip(posx_norm, -1.0, 1.0))
        except Exception:
            posx_norm = 0.0

        return posx_norm, prox_norm

    def reset(self, seed=None, options=None):
        # reset del simulador
        try:
            self.sim.resetSimulation()
        except Exception as e:
            print("Aviso: fallo al resetear simulador:", e)

        # reconectar Robobo por seguridad
        try:
            self.robobo.disconnect()
            self.robobo.connect()
            print("Robobo reconectado")
        except Exception:
            pass

        # Posar la cámara en una posición neutra
        try:
            self.robobo.moveTiltTo(100, 30, wait=True)
        except Exception:
            pass

        self.steps = 0
        # Inicializamos prev_prox (0 = lejos)
        self.prev_prox = 0.0
        self.prev_dist = 1.0  # compatibilidad con código viejo si se usa en otro sitio
        self.trajectory = []

        # Leer observación inicial de forma segura
        try:
            blob = self.robobo.readColorBlob(BlobColor.RED)
        except Exception:
            blob = None

        features = self._get_blob_features(blob)
        if features is None:
            observation = np.array([0.0, 0.0], dtype=np.float32)  # posx 0, prox 0 (lejos)
            self.prev_prox = 0.0
        else:
            posx_norm, prox_norm = features
            observation = np.array([posx_norm, prox_norm], dtype=np.float32)
            self.prev_prox = float(prox_norm)
            self.prev_dist = 1.0 - self.prev_prox

        return observation, {}

    def close(self):
        try:
            self.robobo.disconnect()
        except Exception:
            pass
        try:
            self.sim.disconnect()
        except Exception:
            pass


# --------------------------
# Entrenamiento y evaluación
# --------------------------


def make_env_fn(host="localhost"):
    def _init():
        env = RoboboGymEnv(host=host)
        return env
    return _init


def train(total_timesteps: int = 100000, model_path: str = "ppo_robobo"):
    
    # Creamos y monitorizamos UNA instancia del entorno (no vectorizada)
    env = RoboboGymEnv()
    env = Monitor(env)

    # Debug: confirma el tipo de objeto que SB3 va a usar
    print("DEBUG: env type:", type(env))
    try:
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/robobo_")
        model.learn(total_timesteps=total_timesteps)
        model.save(model_path)
        print(f"Modelo guardado en {model_path}")
    finally:
        try:
            env.close()
        except Exception:
            pass


def evaluate(model_path: str = "ppo_robobo", episodes: int = 5, render: bool = False):
    env = RoboboGymEnv()
    obs, _ = env.reset()

    model = PPO.load(model_path)

    all_trajectories = []

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        traj = []
        total_reward = 0.0
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            # intentar recoger pose si existe
            try:
                if env.trajectory:
                    traj = env.trajectory.copy()
            except Exception:
                pass

        all_trajectories.append(traj)
        print(f"Episodio {ep+1}: recompensa total = {total_reward}")

    # Plot de trayectorias si hay datos
    fig, ax = plt.subplots()
    for t in all_trajectories:
        if len(t) > 0:
            xs, ys = zip(*t)
            ax.plot(xs, ys, '-o', alpha=0.8)
    ax.set_title('Trayectorias (si el simulador provee pose)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    if len(all_trajectories) > 0:
        out = 'trajectories.png'
        fig.savefig(out)
        print(f"Gráfico de trayectorias guardado en {out}")
    else:
        print("No se pudieron obtener trayectorias (simulador no proporcionó poses)")

    env.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'eval'], default='train')
    parser.add_argument('--timesteps', type=int, default=100000)
    parser.add_argument('--model', type=str, default='ppo_robobo')
    parser.add_argument('--episodes', type=int, default=5)
    args = parser.parse_args()

    if args.mode == 'train':
        train(total_timesteps=args.timesteps, model_path=args.model)
    else:
        evaluate(model_path=args.model, episodes=args.episodes)