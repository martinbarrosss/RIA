import gymnasium as gym
from robobopy.Robobo import Robobo
from robobosim.RoboboSim import RoboboSim

sim = RoboboSim("localhost")
sim.connect()

class envRobobo(gym.Env):
    def __init__(self):
        super().__init__()
        self.robobo = Robobo('localhost')
        self.robobo.connect()
        self.action_space = gym.spaces.Discrete(4)  # 4 acciones posibles
        self.observation_space = gym.spaces.Discrete(6)  # 6 estados posibles (podemos empezar con menos)

    def step(self, action):
        # logica para ejecutar una acción y obtener el nuevo estado, recompensa...
        pass

    def reset(self):
        sim.resetSimulation()

    def close(self):
        self.robobo.disconnect()
