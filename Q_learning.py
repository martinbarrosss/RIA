import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Asume que ya tienes las librerías de Robobo instaladas
# Para este código, usaremos 'mock' de las llamadas a Robobo para que sea ejecutable.
# En tu implementación final, debes reemplazar estas llamadas con las reales de RoboboSim.

# from robobo import Robobo
# from robobosim import RoboboSim

# Variables de configuración (puedes ajustarlas según tus necesidades)
MAX_STEPS_PER_EPISODE = 500
TERMINATION_DISTANCE = 0.1 # Metros
FIELD_SIZE = 1.5 # El campo va de -1.5 a 1.5 en x e y

class RoboboEnv(gym.Env):
    """
    Entorno de Robobo que sigue la interfaz de Gymnasium.
    Objetivo: Entrenar a un robot para acercarse a un objeto cilíndrico rojo.
    """
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def _init_(self):
        super(RoboboEnv, self)._init_()

        # --- Definición de los Espacios ---
        # Espacio de observación:
        # Se utilizan espacios continuos (Box) para una mayor complejidad, como se pide.
        # El estado del agente está representado por la distancia y el ángulo al objeto.
        # Las velocidades lineal y angular del robot también pueden ser parte del estado
        # para ayudar al agente a tomar decisiones.
        # [distancia, angulo_relativo, velocidad_lineal_propia, velocidad_angular_propia]
        self.observation_space = spaces.Box(
            low=np.array([0, -np.pi, -100, -100], dtype=np.float32),
            high=np.array([FIELD_SIZE * np.sqrt(2), np.pi, 100, 100], dtype=np.float32),
            dtype=np.float32
        )

        # Espacio de acción:
        # Espacio continuo para las velocidades de las ruedas.
        # [-1.0, 1.0] se mapearán a los rangos de velocidad del robot (e.g., -100 a 100 cm/s).
        # [velocidad_rueda_izquierda, velocidad_rueda_derecha]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Conexión con el simulador RoboboSim.
        # En tu práctica, esto se debería inicializar aquí.
        # self.robobo = Robobo('127.0.0.1')
        # self.robobosim = RoboboSim(self.robobo)

        self.robot_pos = np.array([0.0, 0.0])
        self.object_pos = np.array([0.0, 0.0])
        self.robot_orientation = 0.0 # Orientación en radianes
        self.linear_velocity = 0.0
        self.angular_velocity = 0.0
        self.steps_count = 0

    def _get_obs(self):
        """
        Calcula y devuelve la observación actual del robot.
        """
        # Calcular la distancia y el ángulo al objeto.
        vector_to_object = self.object_pos - self.robot_pos
        distance = np.linalg.norm(vector_to_object)
        
        # Calcular el ángulo relativo al frente del robot.
        angle_to_object = np.arctan2(vector_to_object[1], vector_to_object[0])
        relative_angle = angle_to_object - self.robot_orientation
        
        # Normalizar el ángulo a un rango de -pi a pi
        relative_angle = (relative_angle + np.pi) % (2 * np.pi) - np.pi

        # En tu práctica, esta información provendría de los sensores del robot.
        return np.array([distance, relative_angle, self.linear_velocity, self.angular_velocity], dtype=np.float32)

    def _get_info(self):
        """
        Proporciona información adicional para depuración.
        """
        return {
            "robot_position": self.robot_pos,
            "object_position": self.object_pos,
            "robot_orientation": self.robot_orientation
        }

    def reset(self, seed=None, options=None):
        """
        Reinicia el entorno a un estado inicial aleatorio.
        """
        super().reset(seed=seed)
        
        # Resetear el conteo de pasos para un nuevo episodio
        self.steps_count = 0

        # Posicionar el robot y el objeto en ubicaciones aleatorias
        # dentro de los límites del escenario.
        self.robot_pos = self.np_random.uniform(low=-FIELD_SIZE, high=FIELD_SIZE, size=(2,))
        self.object_pos = self.np_random.uniform(low=-FIELD_SIZE, high=FIELD_SIZE, size=(2,))
        self.robot_orientation = self.np_random.uniform(low=-np.pi, high=np.pi)
        self.linear_velocity = 0.0
        self.angular_velocity = 0.0
        
        # Llama a las funciones del simulador para mover el robot y el objeto.
        # self.robobosim.set_robot_position(x=self.robot_pos[0], y=self.robot_pos[1], rot=self.robot_orientation)
        # self.robobosim.set_cylinder_position(x=self.object_pos[0], y=self.object_pos[1])
        # self.robobo.stop_robot()

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        """
        Avanza el entorno un paso de tiempo aplicando la acción del agente.
        """
        self.steps_count += 1
        
        # Mapear la acción a las velocidades de las ruedas del robot.
        # Puedes usar una simple conversión lineal, por ejemplo, los valores de acción [-1, 1] se
        # convierten en velocidades de rueda de [-100, 100].
        # left_wheel_speed = action[0] * 100
        # right_wheel_speed = action[1] * 100
        # self.robobo.set_motor_speeds(left_wheel_speed, right_wheel_speed)

        # Actualizar la posición del robot basándote en la acción (simulación básica).
        # En la práctica real, el simulador se encargará de esto.
        dt = 0.1 # Paso de tiempo simulado
        self.linear_velocity = (action[0] + action[1]) / 2.0 * 100 # Velocidad lineal de ejemplo
        self.angular_velocity = (action[1] - action[0]) / 2.0 * 100 # Velocidad angular de ejemplo
        self.robot_orientation += self.angular_velocity * dt
        self.robot_pos[0] += self.linear_velocity * np.cos(self.robot_orientation) * dt
        self.robot_pos[1] += self.linear_velocity * np.sin(self.robot_orientation) * dt

        # --- Cálculo de la Recompensa ---
        # Penalización por alejarse del objetivo.
        old_distance = np.linalg.norm(self.object_pos - (self.robot_pos - np.array([self.linear_velocity * np.cos(self.robot_orientation) * dt, self.linear_velocity * np.sin(self.robot_orientation) * dt])))
        current_distance = np.linalg.norm(self.object_pos - self.robot_pos)
        reward = (old_distance - current_distance) * 10.0 # Recompensa por acercarse

        # Recompensa o penalización adicional por la orientación
        reward += (1.0 - abs(self._get_obs()[1])/np.pi) * 0.1 # Recompensa por estar bien orientado

        # Penalización por chocar con los bordes del escenario.
        if abs(self.robot_pos[0]) > FIELD_SIZE or abs(self.robot_pos[1]) > FIELD_SIZE:
            reward -= 5.0 # Penalización severa
            
        # --- Condiciones de finalización del episodio ---
        terminated = current_distance < TERMINATION_DISTANCE
        truncated = self.steps_count >= MAX_STEPS_PER_EPISODE
        
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def render(self):
        # La visualización se manejará con el modo 'human' de Gymnasium,
        # que llamará automáticamente al método render del simulador.
        pass

    def close(self):
        """
        Cierra la conexión con el simulador.
        """
        # self.robobo.stop_robot()
        # self.robobo.disconnect()
        pass

# Ejemplo de uso (no es parte del código de la práctica, solo para probar)
if __name__ == '__main__':
    env = RoboboEnv()
    obs, info = env.reset()
    print("Estado inicial:", obs)
    print("Información inicial:", info)
    
    done = False
    total_reward = 0
    while not done:
        action = env.action_space.sample()  # Acción aleatoria para probar
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        print(f"Paso {env.steps_count}: Acción={action}, Recompensa={reward:.2f}, Distancia={obs[0]:.2f}")
    
    print(f"Episodio terminado. Recompensa total: {total_reward:.2f}")
    env.close()