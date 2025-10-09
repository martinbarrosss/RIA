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
MAX_STEPS = 40
TARGET_BLOB_COLOR = BlobColor.RED
CAMERA_WIDTH = 100
CAMERA_HEIGHT = 100
CENTER_X = CAMERA_WIDTH / 2.0
MAX_WHEEL_SPEED = 15


class RoboboFollowerEnv(gym.Env):

    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None, debug: bool = False, discrete_actions: bool = False):
        
        super().__init__()

        self.logger = logging.getLogger("RoboboFollowerEnv")
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(levelname)s] %(message)s")
        handler.setFormatter(formatter)
        
        if not self.logger.handlers:
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG if debug else logging.INFO)

        self.logger.info(f"Inicializando RoboboFollowerEnv (debug={debug}, discrete_actions={discrete_actions})")

        self.robobo = Robobo('localhost')
        self.sim = RoboboSim('localhost')

        try:
            self.sim.connect()
            self.robobo.connect()
        except Exception as e:
            self.logger.warning(f"Fallo al conectar al Robobo/Sim: {e}")

        self.robobo.wait(1.0)
        self.discrete_actions = discrete_actions

        # --- ACCIONES ---
        # Ahora prohibimos la marcha atrás en las acciones del agente:
        # - Si es discreto: 5 acciones predefinidas (todas con v >= 0)
        # - Si es continuo: v ∈ [0,1] (solo hacia adelante), omega ∈ [-1,1]
        if self.discrete_actions:
            self.action_space = gym.spaces.Discrete(5)  # 5 acciones discretas (todas hacia adelante / giros)
        else:
            # v: [0..1], omega: [-1..1] -> evita que el agente elija velocidades negativas
            self.action_space = gym.spaces.Box(
                low=np.array([0.0, -1.0], dtype=np.float32), 
                high=np.array([1.0, 1.0], dtype=np.float32), 
                dtype=np.float32
            )

        # Observación: [posx_norm, proximity_norm, delta_proximity, blob_visible]
        self.observation_space = gym.spaces.Box(
            low=np.array([-1.0, 0.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        self.steps = 0
        self.max_steps = MAX_STEPS
        self.last_blob_posx = CENTER_X
        self.no_blob_steps = 0

        # nuevo: contar acciones desde la última visión válida del blob
        self.actions_since_seen = 0

        # Diferencia de proximidad anterior (para shaping); None hasta el primer paso/reset
        self.prev_proximity = None

    def _get_observation(self):

        blob = None
        blob_detected = False
        try:
            blob = self.robobo.readColorBlob(TARGET_BLOB_COLOR)
        except Exception as e:
            self.logger.debug(f"(Excepción leyendo blob: {e})")

        MAX_POSY = CAMERA_HEIGHT 
        if blob is not None:
            posx_norm = (blob.posx - CENTER_X) / CENTER_X  
            proximity_norm = blob.posy / MAX_POSY
            obs = np.array([posx_norm, proximity_norm], dtype=np.float32)
            self.logger.debug(f"Blob detectado: posx={blob.posx:.1f}, posy={blob.posy:.1f} -> obs={obs.round(3)}")
            self.actions_since_seen = 0
            blob_detected = True
        else:
            obs = np.array([0.0, 0.0], dtype=np.float32)
            self.actions_since_seen += 1
            self.logger.debug(f"No se detecta blob (contador={self.actions_since_seen}) -> obs=[0.0, 0.0]")

        return obs, blob_detected, self.actions_since_seen

    def _get_reward(self, obs, average_speed, prev_proximity):
        # obs: [posx_norm, proximity_norm, delta_proximity, blob_visible]
        posx_norm = float(obs[0])
        proximity_norm = float(obs[1])

        #  Vamo a darle más importancia a que este centrado
        centering_reward = 1.0 - abs(posx_norm)        # en [0,1]
        centering_reward *= 2.0                        # subir peso

        # proximity reward (suavizada, cuadrática)
        proximity_reward = (proximity_norm ** 2) * 30.0  # cuadrática en lugar de cúbica

        # Premiar avanzar hacia el objetivo, penalizar marcha atrás (no aplica porque agente no puede elegir marcha atrás)
        speed_factor = average_speed / MAX_WHEEL_SPEED
        movement_reward = 0.0
        if proximity_norm > 0.05 and abs(posx_norm) < 0.6:
            movement_reward = speed_factor * centering_reward * 0.5

        # Recompensa por reducción de "distancia" (potential-based shaping)
        potential = proximity_norm
        shaping = 0.0
        if prev_proximity is not None:
            shaping = 10.0 * (potential - prev_proximity)  # positive si nos acercamos

        # combine (ajusta pesos)
        total_reward = 0.5 * centering_reward + 0.4 * proximity_reward + 0.1 * movement_reward + shaping

        # fallbacks / penalties
        if proximity_norm < 0.01:
            return -1.0

        # limit extremes
        total_reward = float(np.clip(total_reward, -20.0, 200.0))

        return total_reward

    def _discrete_to_continuous(self, action):
        """
        Convierte una acción discreta (0-4) en velocidades [izq, der] normalizadas entre -1 y 1.
        Todas las acciones tienen componente v >= 0 (no marcha atrás).
        """
        if action == 0:   # avanzar recto
            return np.array([0.4, 0.4], dtype=np.float32)
        elif action == 1: # girar izquierda brusco (adelante + giro)
            return np.array([0.5, 0.15], dtype=np.float32)
        elif action == 2: # girar izquierda suave
            return np.array([0.4, 0.2], dtype=np.float32)
        elif action == 3: # girar derecha brusco
            return np.array([0.2, 0.4], dtype=np.float32)
        elif action == 4: # girar derecha suave
            return np.array([0.15, 0.5], dtype=np.float32)
        else:
            raise ValueError("Acción discreta fuera de rango")

    def step(self, action):
        self.steps += 1

        # --- LECTURA PREVIA DE IR: detectamos colisión / proximidad antes de ejecutar la acción ---
        try:
            ir_front_pre = self.robobo.readIRSensor(IR.FrontC)
        except Exception as e:
            self.logger.debug(f"Error leyendo IR antes del movimiento: {e}")
            ir_front_pre = 0

        # Si hay una lectura de IR muy alta, ejecutamos maniobra automática de retroceso (excepción)
        if ir_front_pre > 600:
            self.logger.warning(f"Colisión/Proximidad detectada por IR antes de ejecutar acción (IR={ir_front_pre}). Ejecutando retroceso de emergencia.")
            try:
                # maniobra de retroceso automática: ruedas en negativo (único caso donde se usa marcha atrás)
                back_speed = -0.5 * MAX_WHEEL_SPEED
                self.robobo.moveWheelsByTime(back_speed, back_speed, 0.6, wait=True)
            except Exception as e:
                self.logger.warning(f"Error ejecutando retroceso de emergencia: {e}")

            # dar la recompensa máxima/clipping y terminar episodio (o podrías decidir no terminar)
            reward = 200.0
            terminated = True
            truncated = False
            info = {"steps": self.steps, "ir": ir_front_pre, "actions_since_seen": self.actions_since_seen, "emergency_back": True}
            # Actualizar prev_proximity a 0 para reflejar que se alejó
            self.prev_proximity = 0.0
            return self._get_observation(), reward, terminated, truncated, info

        # Si es discreto, convertir a continua usando el mapeo definido
        if self.discrete_actions:
            self.logger.info(f"Paso {self.steps}: Acción discreta={action}")
            action = self._discrete_to_continuous(action)

        # A partir de aquí asumimos acción continua en forma [v, omega]
        try:
            v = float(action[0])
            omega = float(action[1])
        except Exception:
            # Si action viene en otra forma, intentar convertir a numpy array
            action = np.asarray(action, dtype=np.float32)
            v = float(action[0])
            omega = float(action[1])

        # clip: ahora v está forzado a ser >= 0 por el action_space, pero por seguridad lo clippeamos
        v = float(np.clip(v, 0.0, 1.0))
        omega = float(np.clip(omega, -1.0, 1.0))

        self.logger.info(f"Paso {self.steps}: Acción continua [v,omega]=[{v:.3f},{omega:.3f}]")

        # Mapeo lineal sencillo a velocidades de ruedas
        vel_izq = (v - omega) * MAX_WHEEL_SPEED
        vel_der = (v + omega) * MAX_WHEEL_SPEED

        # velocidad promedio (valor absoluto longitudinal aproximado) para la recompensa
        average_speed = abs(v) * MAX_WHEEL_SPEED

        self.logger.info(f"Velocidades: izq={vel_izq:.1f}, der={vel_der:.1f}")

        try:
            self.robobo.moveWheelsByTime(vel_izq, vel_der, 0.2, wait=True)
        except Exception as e:
            self.logger.warning(f"Error ejecutando movimiento: {e}")

        obs, blob_detected = self._get_observation()
        # llamar _get_reward con prev_proximity para el shaping
        reward = self._get_reward(obs, average_speed, self.prev_proximity)

        # actualizar prev_proximity para el siguiente paso (usar proximity actual)
        try:
            proximity_norm = float(obs[1])
            self.prev_proximity = proximity_norm
        except Exception:
            self.prev_proximity = None
            proximity_norm = obs[1]

        terminated = False

        if proximity_norm > 0.975:
            reward += 100.0
            terminated = True
            self.logger.info(f"Éxito: objeto muy cerca (proximity={proximity_norm:.3f}). Episodio terminado.")

        # Leer IR una vez por paso (post-movimiento para info adicional)
        try:
            ir_c = self.robobo.readIRSensor(IR.FrontC)
        except Exception as e:
            self.logger.debug(f"Error leyendo IR en step: {e}")
            ir_c = 0

        # Si la lectura post-movimiento indica colisión, actuamos igual que en la lectura previa:
        if blob_detected and ir_c > 600 and not terminated:
            self.logger.warning(f"IR > 600 después del movimiento ({ir_c}). Ejecutando retroceso de emergencia.")
            try:
                back_speed = -0.5 * MAX_WHEEL_SPEED
                self.robobo.moveWheelsByTime(back_speed, back_speed, 0.6, wait=True)
            except Exception as e:
                self.logger.warning(f"Error ejecutando retroceso de emergencia: {e}")
            reward = 200.0
            terminated = True

        # Si no ve el blob durante N acciones, penalizar y terminar (reset)
        if not blob_detected and self.actions_since_seen >= 3:
            reward -= 20.0  # penalización fuerte para desalentar perder el objetivo
            terminated = True
            self.logger.warning("No se vio el objeto en 5 acciones consecutivas -> penalización y fin del episodio.")

        # Mantener tu lógica previa de no_blob_steps si quieres también
        if self.no_blob_steps >= 3 and not terminated:
            reward -= 5.0 
            terminated = True
            self.logger.warning("No se detectó el cilindro en 5 pasos (no_blob_steps), episodio terminado")

        truncated = self.steps >= self.max_steps
        info = {"steps": self.steps, "ir": ir_c, "actions_since_seen": self.actions_since_seen}

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
        # reiniciar prev_proximity al reset
        self.prev_proximity = None
        # reiniciar contador de no_blob y acciones_sin_ver
        self.no_blob_steps = 0
        self.actions_since_seen = 0

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
