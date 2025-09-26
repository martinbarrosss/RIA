# import gymnasium as gym
# from robobopy.Robobo import Robobo
# from robobosim.RoboboSim import RoboboSim

# class envRobobo(gym.Env):
#     def __init__(self):
#         super().__init__()
#         self.robobo = Robobo('localhost')
#         self.robobo.connect()
#         self.sim = RoboboSim("localhost")
#         self.sim.connect()
#         self.action_space = gym.spaces.Discrete(4)  # 4 acciones posibles
#         self.observation_space = gym.spaces.Box(low=0, high=255, shape=(8,), dtype=int)  # 8 IR sensores

#     def step(self, action):
#         # Ejecuta la acción en el robot
#         if action == 0:
#             self.robobo.moveWheels(25, 25)
#         elif action == 1:
#             self.robobo.moveWheels(-25, -25)
#         elif action == 2:
#             self.robobo.moveWheels(-5, 15)
#         elif action == 3:
#             self.robobo.moveWheels(15, -5)
#         self.sim.wait(1)

#         # Observa estado
#         obs = (self.robobo.readAllIRSensor())
#         print(obs)
#         # Recompensa positiva si ningún sensor detecta obstáculo cerca
#         # reward = 1 if min(obs) > 100 else -10
#         # terminated = min(obs) <= 20  # Termina episodio si muy cerca de obstáculo
#         # return obs, reward, terminated, False, {"info": "estado"}


#     def reset(self, seed=None, options=None):
#         self.sim.resetSimulation()
#         # Supón que obtienes la observación con robobo.readAllIRSensor()
#         observation = self.robobo.readAllIRSensor()
#         return observation, {}


#     def close(self):
#         self.robobo.disconnect()