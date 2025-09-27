from envRobobo import envRobobo
import time

env = envRobobo()

obs, info = env.reset()
print("observación inicial:", obs)

for step in range(50):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Paso {step+1}: acción={action}, obs={obs}, reward={reward}, terminated={terminated}, truncated={truncated}")
    env.robobo.wait(0.5)

    if terminated or truncated:
        print("episodio finalizado. Reiniciando enctorno\n")
        obs, info = env.reset()

env.close()
