# robobo_env_debug.py

import gymnasium as gym
import numpy as np
from robobopy.Robobo import Robobo
from robobosim.RoboboSim import RoboboSim
from robobopy.utils.BlobColor import BlobColor
from robobopy.utils.IR import IR 
from robobopy.utils.Wheels import Wheels
import time
import logging

# Parámetros del entorno
MAX_STEPS = 500
TARGET_BLOB_COLOR = BlobColor.RED
CAMERA_WIDTH = 100
CAMERA_HEIGHT = 100
CENTER_X = CAMERA_WIDTH / 2.0
MAX_WHEEL_SPEED = 80


class RoboboFollowerEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None, debug: bool = False):
        super().__init__()

        self.logger = logging.getLogger("RoboboFollowerEnv")
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(levelname)s] %(message)s")
        handler.setFormatter(formatter)
        if not self.logger.handlers:
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG if debug else logging.INFO)

        self.logger.info("Inicializando RoboboFollowerEnv (debug=%s)" % debug)

        self.robobo = Robobo('localhost')
        self.sim = RoboboSim('localhost')

        try:
            self.sim.connect()
            self.robobo.connect()
        except Exception as e:
            self.logger.warning(f"Fallo al conectar al Robobo/Sim: {e}")

        time.sleep(1.0)

        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32), 
            high=np.array([1.0, 1.0], dtype=np.float32), 
            dtype=np.float32
        )
        
        self.observation_space = gym.spaces.Box(
            low=np.array([-1.0, 0.0], dtype=np.float32), 
            high=np.array([1.0, 1.0], dtype=np.float32), 
            dtype=np.float32
        )

        self.steps = 0
        self.max_steps = MAX_STEPS
        self.last_blob_posx = CENTER_X

    def _get_observation(self):
        blob = None
        try:
            blob = self.robobo.readColorBlob(TARGET_BLOB_COLOR)
        except Exception as e:
            self.logger.debug(f"Excepción leyendo blob: {e}")

        MAX_POSY = CAMERA_HEIGHT 
        if blob is not None:
            posx_norm = (blob.posx - CENTER_X) / CENTER_X  
            proximity_norm = blob.posy / MAX_POSY
            obs = np.array([posx_norm, proximity_norm], dtype=np.float32)
            self.logger.debug(f"Blob detectado: posx={blob.posx:.1f}, posy={blob.posy:.1f} -> obs={obs.round(3)}")
        else:
            obs = np.array([0.0, 0.0], dtype=np.float32)
            self.logger.debug("No se detecta blob: obs=[0.0, 0.0]")
        return obs

    def _get_reward(self, obs, average_speed): 
        posx_norm, proximity_norm = obs 
        centering_reward = 1.0 - abs(posx_norm)
        proximity_reward = (proximity_norm ** 3) * 20.0
        speed_factor = average_speed / MAX_WHEEL_SPEED
        movement_reward = 0.0
        if proximity_norm > 0.1 and abs(posx_norm) < 0.5: 
            movement_reward = speed_factor * centering_reward * 1.0 

        if proximity_norm < 0.01:
            self.logger.debug("Recompensa: objeto no detectado -> penalización -1.0")
            return -1.0

        ir_c = 0
        try:
            ir_c = self.robobo.readIRSensor(IR.FrontC)
        except Exception as e:
            self.logger.debug(f"Error leyendo IR: {e}")

        if ir_c > 800: 
            self.logger.warning(f"IR alto detectado (FrontC={ir_c}) -> Colisión probable. Recompensa -10")
            return -10.0

        total_reward = (centering_reward * 0.1) + (proximity_reward * 0.7) + (movement_reward * 0.2)
        self.logger.debug(
            f"Reward -> centering: {centering_reward:.3f}, proximity: {proximity_reward:.3f}, movement: {movement_reward:.3f}, total: {total_reward:.3f}"
        )
        return total_reward

    def step(self, action):
        self.steps += 1
        action = np.clip(action, self.action_space.low, self.action_space.high)

        vel_izq = float(action[0]) * MAX_WHEEL_SPEED
        vel_der = float(action[1]) * MAX_WHEEL_SPEED
        average_speed = (abs(vel_izq) + abs(vel_der)) / 2.0 

        self.logger.info(f"Paso {self.steps}: Acción={action.round(3)}, vel_izq={vel_izq:.1f}, vel_der={vel_der:.1f}")

        try:
            self.robobo.moveWheelsByTime(vel_izq, vel_der, 0.5, wait=False)
            self.robobo.wait(0.5)  # Espera ampliada de 0.3 a 0.5 segundos
            time.sleep(0.5)        # Pausa adicional para dar tiempo al simulador
        except Exception as e:
            self.logger.warning(f"Error ejecutando movimiento: {e}")

        obs = self._get_observation()
        reward = self._get_reward(obs, average_speed)

        proximity_norm = obs[1]
        terminated = False

        if proximity_norm > 0.95:
            reward += 100.0
            terminated = True
            self.logger.info(f"Éxito: objeto muy cerca (proximity={proximity_norm:.3f}). Episodio terminado.")

        if reward < -8:
            terminated = True
            self.logger.warning("Terminado por recompensa baja (probable colisión o fallo crítico).")

        truncated = self.steps >= self.max_steps
        info = {"steps": self.steps}

        if reward < -8:
            truncated = True

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.logger.info("Reset del entorno: reiniciando simulación y sensores...")

        try:
            self.sim.resetSimulation()
        except Exception as e:
            self.logger.warning(f"Error reseteando simulador: {e}")

        time.sleep(1.5)

        try:
            self.robobo.moveTiltTo(100, 100, wait=True)
            self.robobo.movePanTo(0, 100, wait=True)
            time.sleep(0.8)

            ir_c = 0
            try:
                ir_c = self.robobo.readIRSensor(IR.FrontC)
            except Exception as e:
                self.logger.debug(f"Error leyendo IR en reset: {e}")

            if ir_c > 600:
                 self.logger.warning("ADVERTENCIA: Iniciando cerca de colisión. Retrocediendo...")
                 try:
                     self.robobo.moveWheelsByTime(-20, -20, 0.6, wait=True)
                 except Exception as e:
                     self.logger.warning(f"Error retrocediendo durante reset: {e}")
                 time.sleep(0.6)

        except Exception as e:
            self.logger.warning(f"Error al configurar Robobo/Sim durante reset: {e}. Reintentando conexión.")
            try:
                self.robobo.disconnect()
                self.robobo.connect()
                time.sleep(1.0)
                self.robobo.moveTiltTo(100, 100, wait=True)
                self.robobo.movePanTo(0, 100, wait=True)
            except Exception as e2:
                self.logger.error(f"Reconexión fallida: {e2}")

        self.steps = 0
        observation = self._get_observation()
        info = {}
        self.logger.info(f"Reset completo. Observación inicial: {observation.round(3)}")
        return observation, info

    def close(self):
        self.logger.info("Cerrando conexiones Robobo/Sim...")
        try:
            self.robobo.disconnect()
        except Exception:
            pass
        try:
            self.sim.disconnect()
        except Exception:
            pass

if __name__ == '__main__':
    
    env = RoboboFollowerEnv(debug=True)
    obs, info = env.reset()
    print("Observación inicial:", obs)

    for step in range(200):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Paso {step+1}: Obs={obs.round(2)}, Reward={reward:.2f}, Terminated={terminated}, Truncated={truncated}")
        if terminated or truncated:
            print("Episodio finalizado. Reiniciando entorno\n")
            obs, info = env.reset()
            env.robobo.wait(1)
    env.close()
