from robobopy.Robobo import Robobo
from robobosim.RoboboSim import RoboboSim

robobo = Robobo("localhost")
robobo.connect()

sim = RoboboSim("localhost")
sim.connect()