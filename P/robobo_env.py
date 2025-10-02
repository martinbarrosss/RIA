# robobo_env.py

import gymnasium as gym
import numpy as np
from robobopy.Robobo import Robobo
from robobosim.RoboboSim import RoboboSim
from robobopy.utils.BlobColor import BlobColor
from robobopy.utils.IR import IR 
from robobopy.utils.Wheels import Wheels
import time

# Parámetros del entorno
MAX_STEPS = 500  # Aumentamos los pasos para un RL profundo
TARGET_BLOB_COLOR = BlobColor.RED
CAMERA_WIDTH = 100  # Pixels
CAMERA_HEIGHT = 100 # Pixels
CENTER_X = CAMERA_WIDTH / 2.0
MAX_WHEEL_SPEED = 100 # Velocidad máxima para los motores de Robobo

class RoboboFollowerEnv(gym.Env):
    """
    Entorno de Gymnasium para entrenar a Robobo a seguir un objeto (cilindro rojo).
    Utiliza espacios continuos para acción y observación.
    """
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()
        
        # Conexión con Robobo y Simulador
        self.robobo = Robobo('localhost')
        self.sim = RoboboSim('localhost')

        # Conectar. Añadimos un pequeño wait para asegurarnos
        self.sim.connect()
        self.robobo.connect() 
        # Añade un pequeño delay inicial para que el simulador "se asiente"
        time.sleep(1.0) 

        # --- 1. Espacio de Acciones (Continuo)  ---
        # Acción: [velocidad_rueda_izq, velocidad_rueda_der]
        # Límites: [-MAX_WHEEL_SPEED, MAX_WHEEL_SPEED]
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32), 
            high=np.array([1.0, 1.0], dtype=np.float32), 
            dtype=np.float32 # PPO prefiere acciones normalizadas
        )
        
        # --- 2. Espacio de Observaciones (Continuo) ---
        # Observación: [posición_x_normalizada, proximidad_normalizada (posy)]
        # Rango X: -1.0 a 1.0 (centrado en 0)
        # Rango Proximidad (Y): 0.0 a 1.0 (0 lejos, 1.0 cerca)
        self.observation_space = gym.spaces.Box(
            low=np.array([-1.0, 0.0], dtype=np.float32), 
            high=np.array([1.0, 1.0], dtype=np.float32), 
            dtype=np.float32
        )

        self.steps = 0
        self.max_steps = MAX_STEPS
        self.last_blob_posx = CENTER_X # Para calcular el cambio de posición

    def _get_observation(self):

        """Lee el sensor y normaliza la observación."""
        blob = self.robobo.readColorBlob(TARGET_BLOB_COLOR)
        
        # Máxima posy posible (asumiendo que 100 es el borde inferior)
        MAX_POSY = CAMERA_HEIGHT 
        
        if blob is not None:
            # Normalizar Posición X: de [0, 100] a [-1.0, 1.0] (Centrado)
            posx_norm = (blob.posx - CENTER_X) / CENTER_X  
            
            # Normalizar Proximidad (usando posy): de [0, 100] a [0.0, 1.0]
            # Si posy es 100 (cerca), proximity_norm es 1.0
            # Si posy es 0 (lejos), proximity_norm es 0.0
            proximity_norm = blob.posy / MAX_POSY
            
            # Observación: [posición_x_normalizada, proximidad_normalizada (posy)]
            obs = np.array([posx_norm, proximity_norm], dtype=np.float32)
        else:
            # Si no hay blob, la observación es que está fuera y la proximidad es 0.
            obs = np.array([0.0, 0.0], dtype=np.float32) 
            
        return obs

    def _get_reward(self, obs, average_speed): 
        posx_norm, proximity_norm = obs 
        
        # 1. Recompensa de Centrado
        centering_reward = 1.0 - abs(posx_norm) 

        # 2. RECOMPENSA DE PROXIMIDAD: Aumentar el factor y usar potencia.
        # proximity_norm está entre 0.0 y 1.0
        # proximity_norm ** 3 hace que el valor solo sea alto cuando está MUY cerca (ej. 0.9^3 = 0.729)
        # Multiplicamos por un factor alto (ej. 20.0) para que la recompensa total sea positiva.
        proximity_reward = (proximity_norm ** 3) * 20.0 # Máx. 20.0

        # 3. Recompensa por Velocidad (ahora es más sensible)
        speed_factor = average_speed / MAX_WHEEL_SPEED
        movement_reward = 0.0
        
        # Premiamos la velocidad SOLO si está relativamente bien orientado.
        if proximity_norm > 0.1 and abs(posx_norm) < 0.5: 
            # Recompensa el movimiento rápido y centrado.
            movement_reward = speed_factor * centering_reward * 1.0 
        
        # 4. Penalización por Ausencia del Objeto (Mantenemos suave)
        if proximity_norm < 0.01:
            return -1.0 

        # 5. Penalización por Colisión (Máximo castigo)
        ir_c = self.robobo.readIRSensor(IR.FrontC)
        if ir_c > 800: 
            return -10.0 

        # Recompensa total: Ajustamos los pesos. El peso del centrado puede bajar.
        total_reward = (centering_reward * 0.1) + (proximity_reward * 0.7) + (movement_reward * 0.2) 
        
        return total_reward

    def step(self, action):

        # 1. Escalar y asegurar el rango
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Escalar las acciones normalizadas [-1, 1] a la velocidad real [-MAX_WHEEL_SPEED, MAX_WHEEL_SPEED]
        vel_izq = action[0] * MAX_WHEEL_SPEED
        vel_der = action[1] * MAX_WHEEL_SPEED
        
        # Guardar la velocidad media para usarla en la recompensa
        average_speed = (abs(vel_izq) + abs(vel_der)) / 2.0 
        
        # 2. Ejecutar acción
        self.robobo.moveWheelsByTime(vel_izq, vel_der, 0.3, wait=False)
        self.robobo.wait(0.3) 
        
        # 3. Obtener estado y recompensa
        obs = self._get_observation()
        reward = self._get_reward(obs, average_speed) 
        
        # Obtener la proximidad normalizada para la condición de éxito
        proximity_norm = obs[1] 

        # Condición de Truncado/Terminado (reset)
        terminated = False # Usaremos 'terminated' para éxito/colisión

        # A. ÉXITO (Condición de Tarea Cumplida)
        if proximity_norm > 0.95: 
            reward += 100.0 # Recompensa ENORME por éxito
            terminated = True # Éxito termina el episodio

        # B. Colisión (Condición de Fracaso, ya manejada en _get_reward, pero la repetimos para claridad de terminación)
        if reward < -8: 
            terminated = True 

        # El truncado solo ocurre por límite de tiempo
        truncated = self.steps >= 500

        info = {"steps": self.steps}
        
        # Si la recompensa es muy baja (ej. colisión), forzamos un reset.
        if reward < -8:
            truncated = True # O terminated = True, dependiendo de la política

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)
        
        # 1. Reiniciar la simulación completamente
        self.sim.resetSimulation() 
        time.sleep(1.0) # Aumentar la espera después de resetear la simulación (para estabilidad)

        # 2. Reestablecer la cámara y asegurar una posición inicial segura
        try:
            # Configuración de cámara con velocidad alta (100) para un movimiento rápido y estable
            self.robobo.moveTiltTo(100, 100, wait=True) 
            self.robobo.movePanTo(0, 100, wait=True) 
            time.sleep(0.5) # Pausa después de mover la cámara

            # --- CHEQUEO DE SEGURIDAD (IR ALTA = CERCA/PELIGRO) ---
            ir_c = self.robobo.readIRSensor(IR.FrontC)
            
            # Si el valor IR es alto (> 600), el robot está muy cerca de algo al empezar.
            # Lo forzamos a retroceder suavemente.
            if ir_c > 600: 
                 print("ADVERTENCIA: Iniciando cerca de colisión (IR alto). Retrocediendo...")
                 self.robobo.moveWheelsByTime(-20, -20, 0.5, wait=True) # Retrocede
                 time.sleep(0.5) # Espera a que termine el movimiento
            # --- FIN CHEQUEO ---

        except Exception as e:
            # Este bloque maneja la pérdida de conexión si ocurre durante el reset
            print(f"Error al configurar Robobo/Sim: {e}. Reintentando conexión.")
            self.robobo.disconnect()
            self.robobo.connect()
            time.sleep(1.0) # Espera extra
            self.robobo.moveTiltTo(100, 100, wait=True)
            self.robobo.movePanTo(0, 100, wait=True)

        self.steps = 0
        observation = self._get_observation()
        
        info = {}

        return observation, info

    def close(self):
        self.robobo.disconnect()
        self.sim.disconnect()


if __name__ == '__main__':

    # Ejemplo de uso del entorno
    env = RoboboFollowerEnv()
    obs, info = env.reset()
    print("Observación inicial:", obs)

    # El robot realizará 200 pasos con acciones aleatorias
    for step in range(200):
        # Muestra cómo se generan acciones continuas: [vel_izq, vel_der]
        action = env.action_space.sample() 
        
        # Forzamos al robot a ir hacia delante/girar suavemente en la demostración
        # action = np.array([50, 50], dtype=np.float32)
        
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Paso {step+1}: Acción={action.round(1)}, Obs={obs.round(2)}, Reward={reward:.2f}, Terminated={terminated}, Truncated={truncated}")
        
        if terminated or truncated:
            print("Episodio finalizado. Reiniciando entorno\n")
            obs, info = env.reset()
            env.robobo.wait(1) # Espera extra para ver el reset
            
    env.close()