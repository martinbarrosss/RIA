from robobopy.Robobo import Robobo
import gymnasium as gym
#from envRobobo import envRobobo
from robobosim.RoboboSim import RoboboSim
from robobopy.utils.IR import IR


# robobo = Robobo('localhost')
# robobo.connect()
# sim = RoboboSim("localhost")
# sim.connect()
# env = envRobobo()
# obs, info = env.reset()
# total_reward = 0
# done = False
# while not done:
#     action = env.action_space.sample()
#     obs = (robobo.readAllIRSensor())
#     print(obs)
#     # obs, reward, terminated, truncated, info = env.step(action)
#     # total_reward += reward
#     # done = terminated or truncated
# print(f"episodio acabado, recompensa total: {total_reward}")
# env.close()

from robobopy.Robobo import Robobo
import time

robobo = Robobo('localhost')
robobo.connect()


def main():
    robobo.moveWheelsByTime(30, 30, 0.5)
    irs = robobo.readAllIRSensor()
    if irs:
        for sensor_name, sensor_value in irs.items():
            print(f"Sensor {sensor_name}: {sensor_value}")
    else:
        print("No hay datos de sensores IR disponibles")
main()


