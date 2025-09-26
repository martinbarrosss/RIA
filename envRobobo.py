import gymnasium as gym
import numpy as np
from robobopy.Robobo import Robobo
from robobosim.RoboboSim import RoboboSim
from robobopy.utils.BlobColor import BlobColor
from robobopy.utils.Wheels import Wheels


class envRobobo(gym.Env):
    def __init__(self):
        super().__init__()
        self.robobo = Robobo('localhost')
        self.robobo.connect()
        self.sim = RoboboSim("localhost")
        self.sim.connect()

        # Espacios
        self.action_space = gym.spaces.Discrete(4)  # 4 acciones posibles
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(1,), dtype=np.float32)  # 1 sensor

        # Contador de pasos (para truncado)
        self.steps = 0
        self.max_steps = 50

    def step(self, action):
        # Ejecuta la acción
        if action == 0:
            self.robobo.moveWheelsByTime(25, 25, 1.5)
        elif action == 1:
            self.robobo.moveWheelsByTime(-5, 25, 1.5)  # giro derecha
        elif action == 2:
            self.robobo.moveWheelsByTime(25, -5, 1.5)  # giro izquierda
        elif action == 3:
            self.robobo.moveWheelsByDegrees(Wheels.Both, 180, 25)
        self.sim.wait(1)

        # Observa estado (maneja el caso de que no haya blob)
        blob = self.robobo.readColorBlob(BlobColor.RED)
        if blob is not None:
            obs = np.array([blob.posx], dtype=np.float32)
        else:
            obs = np.array([0.0], dtype=np.float32)  # valor por defecto si no hay blob

        print("Observación:", obs[0])

        # Calcular recompensa
        if blob is not None:
            reward = 1 - abs(50 - blob.posx) / 50
        else:
            reward = -1

        # Terminar episodio si se cumple condición
        terminated = reward > 0.9

        # Incrementar contador y truncar si supera el máximo
        self.steps += 1
        truncated = self.steps >= self.max_steps

        info = {}
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        self.sim.resetSimulation()
        self.steps = 0

        # Leer observación inicial de forma segura
        blob = self.robobo.readColorBlob(BlobColor.RED)
        if blob is not None:
            observation = np.array([blob.posx], dtype=np.float32)
        else:
            observation = np.array([0.0], dtype=np.float32)

        return observation, {}

    def close(self):
        self.robobo.disconnect()
