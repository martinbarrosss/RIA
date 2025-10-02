# run_policy.py

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from robobo_env import RoboboFollowerEnv

MODEL_PATH = "./robobo_models/robobo_ppo_final.zip"
EPISODES = 5 # Número de episodios para la validación
STEPS_PER_EPISODE = 500

def validate_and_plot_policy():
    # Cargar el entorno
    env = RoboboFollowerEnv()
    
    # Cargar la política aprendida [cite: 62]
    try:
        model = PPO.load(MODEL_PATH)
        print(f"Política cargada correctamente desde {MODEL_PATH}")
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        print("Asegúrate de ejecutar primero train_ppo.py")
        env.close()
        return

    all_positions = [] # Para almacenar las coordenadas (X, Y) del robot
    
    for episode in range(EPISODES):
        obs, info = env.reset()
        
        # Almacenar las posiciones 2D del robot para el plot
        current_positions = []
        
        for step in range(STEPS_PER_EPISODE):
            # Obtener posición real de Robobo (función exclusiva de RoboboSim)
            try:
                # Obtenemos la posición (X, Y) en el sistema de coordenadas del simulador
                robobo_pos = env.sim.getRoboboPos() 
                current_positions.append((robobo_pos.x, robobo_pos.z)) 
            except Exception as e:
                # Esto puede fallar si RoboboSim no está en un estado válido
                pass
            
            # El agente toma una acción basada en la política
            action, _ = model.predict(obs, deterministic=True) 
            
            # Ejecutar paso
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                break
                
        all_positions.append(current_positions)
        print(f"Episodio {episode+1} finalizado. Pasos: {info.get('steps', STEPS_PER_EPISODE)}")

    env.close()
    
    # --- Representación en un plano 2D  ---
    plt.figure(figsize=(10, 8))
    for i, positions in enumerate(all_positions):
        x = [p[0] for p in positions]
        y = [p[1] for p in positions]
        plt.plot(x, y, label=f'Trayectoria Episodio {i+1}', alpha=0.7)

    # Marcar el punto inicial (asumimos que el reset lo coloca cerca de (0,0) o lo inicializa)
    if all_positions and all_positions[0]:
        plt.plot(all_positions[0][0][0], all_positions[0][0][1], 'go', label='Inicio') 
    
    plt.title('Trayectoria 2D del Robobo (Validación de Política)')
    plt.xlabel('Posición X (Simulador)')
    plt.ylabel('Posición Y (Simulador)')
    plt.grid(True)
    plt.legend()
    plt.savefig('robobo_trajectories.png') # Guardar la figura [cite: 64]
    plt.show()

if __name__ == '__main__':
    validate_and_plot_policy()