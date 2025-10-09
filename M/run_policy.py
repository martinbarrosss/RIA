# run_policy.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from robobo_env import RoboboFollowerEnv

MODEL_PATH = "./robobo_models/robobo_ppo_final.zip"
EPISODES = 5
STEPS_PER_EPISODE = 500

def validate_and_plot_policy():
    # --- Cargar entorno ---
    env = RoboboFollowerEnv()

    # --- Cargar modelo ---
    try:
        model = PPO.load(MODEL_PATH)
        print(f"✅ Política cargada desde {MODEL_PATH}")
    except Exception as e:
        print(f"❌ Error al cargar modelo: {e}")
        env.close()
        return

    all_positions = []      # Lista de trayectorias (X, Z)
    rewards_total = []      # Recompensa total por episodio
    rewards_mean = []       # Recompensa media por episodio

    # --- Ejecutar episodios ---
    for episode in range(EPISODES):
        obs, info = env.reset()
        total_reward = 0
        steps_done = 0
        current_positions = []

        for step in range(STEPS_PER_EPISODE):
            # Guardar posición X, Z
            try:
                robobo_state = env.sim.getRobotLocation(0)
                if robobo_state and "position" in robobo_state:
                    x = robobo_state["position"]["x"]
                    z = robobo_state["position"]["z"]
                    current_positions.append((x, z))
            except Exception as e:
                print("⚠️ No se pudo obtener posición:", e)

            # Predecir acción
            action, _ = model.predict(obs, deterministic=True)

            # Ejecutar paso
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps_done += 1

            if terminated or truncated:
                break

        # Guardar métricas por episodio
        mean_reward = total_reward / steps_done if steps_done > 0 else 0
        rewards_total.append(total_reward)
        rewards_mean.append(mean_reward)
        all_positions.append(current_positions)

        print(f"📊 Episodio {episode+1} | Pasos: {steps_done} | Recompensa total: {total_reward:.2f} | Media: {mean_reward:.2f}")

    env.close()

    # --- 📍 Graficar trayectorias 2D ---
    plt.figure(figsize=(10, 8))
    for i, positions in enumerate(all_positions):
        if len(positions) > 0:
            x_vals = [p[0] for p in positions]
            z_vals = [p[1] for p in positions]
            plt.plot(x_vals, z_vals, label=f"Episodio {i+1}", alpha=0.7)
        else:
            print(f"⚠️ Episodio {i+1} no tiene posiciones registradas.")

    if all_positions and all_positions[0]:
        plt.plot(all_positions[0][0][0], all_positions[0][0][1], 'go', label='Inicio')

    plt.title("📍 Trayectoria 2D del Robobo")
    plt.xlabel("Posición X")
    plt.ylabel("Posición Z")
    plt.grid(True)
    plt.legend()
    plt.savefig("robobo_trajectories.png", dpi=300)
    plt.show()

    # --- 📊 Graficar recompensas con seaborn ---
    df = pd.DataFrame({
        "Episodio": np.arange(1, EPISODES + 1),
        "Recompensa Total": rewards_total,
        "Recompensa Media": rewards_mean
    })

    sns.set(style="whitegrid", font_scale=1.2)

    # Total reward
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x="Episodio", y="Recompensa Total", color="royalblue")
    plt.title("📈 Recompensa total por episodio")
    plt.ylabel("Recompensa total")
    plt.savefig("robobo_rewards_total.png", dpi=300)
    plt.show()

    # Mean reward
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=df, x="Episodio", y="Recompensa Media", marker="o", color="green")
    plt.title("📊 Recompensa media por episodio")
    plt.ylabel("Recompensa media")
    plt.savefig("robobo_rewards_mean.png", dpi=300)
    plt.show()

    print("✅ Gráficos guardados: robobo_trajectories.png, robobo_rewards_total.png, robobo_rewards_mean.png")


if __name__ == '__main__':
    validate_and_plot_policy()
