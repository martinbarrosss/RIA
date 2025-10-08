from robobopy.Robobo import Robobo

robobo = Robobo('localhost')
robobo.connect()

robobo.moveWheelsByTime(20, 60, 2.0)  # Mover hacia adelante durante 2 segundos