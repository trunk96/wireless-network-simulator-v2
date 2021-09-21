from wns2.basestation.nrbasestation import NRBaseStation
from wns2.basestation.dronebasestation import DroneBaseStation
from wns2.basestation.satellitebasestation import SatelliteBaseStation
from wns2.userequipment.userequipment import UserEquipment
from wns2.userequipment.multipath_userequipment import MultiPathUserEquipment
from wns2.environment.environment import Environment
from wns2.environment.environment import MIN_RSRP
from wns2.renderer.renderer import CustomRenderer
import numpy.random as random
import logging

logger = logging.getLogger()
logger.setLevel(level=logging.INFO)


x_lim = 1000
y_lim = 1000
env = Environment(x_lim, y_lim, renderer = CustomRenderer())
MIN_RSRP = -70

for i in range(0, 50):
    pos = (random.rand()*x_lim, random.rand()*y_lim, 1)
    env.add_user(MultiPathUserEquipment(env, i, 25, pos, speed = 0, direction = random.randint(0, 360), _lambda_c=5, _lambda_d = 15))

bs_parm =[{"pos": (500, 500, 30),
    "freq": 800,
    "numerology": 1, 
    "power": 20,
    "gain": 16,
    "loss": 3,
    "bandwidth": 20,
    "max_bitrate": 1000},
    
    #BS2
    {"pos": (250, 300, 30),
    "freq": 1700,
    "numerology": 1, 
    "power": 20,
    "gain": 16,
    "loss": 3,
    "bandwidth": 40,
    "max_bitrate": 1000},

    #BS3
    {"pos": (500, 125, 30),
    "freq": 1900,
    "numerology": 1, 
    "power": 20,
    "gain": 16,
    "loss": 3,
    "bandwidth": 40,
    #15
    "max_bitrate": 1000},

    #BS4
    {"pos": (750, 300, 30),
    "freq": 2000,
    "numerology": 1, 
    "power": 20,
    "gain": 16,
    "loss": 3,
    "bandwidth": 25,
    "max_bitrate": 1000},
    
    #BS5
    {"pos": (750, 700, 30),
    "freq": 1700,
    "numerology": 1, 
    "power": 20,
    "gain": 16,
    "loss": 3,
    "bandwidth": 40,
    "max_bitrate": 1000},

    #BS6
    {"pos": (500, 875, 30),
    "freq": 1900,
    "numerology": 1, 
    "power": 20,
    "gain": 16,
    "loss": 3,
    "bandwidth": 40,
    "max_bitrate": 1000},

    #BS7
    {"pos": (250, 700, 30),
    "freq": 2000,
    "numerology": 1, 
    "power": 20,
    "gain": 16,
    "loss": 3,
    "bandwidth": 25,
    "max_bitrate": 1000}]

for i in range(len(bs_parm)):
    env.add_base_station(NRBaseStation(env, i, bs_parm[i]["pos"], bs_parm[i]["freq"], bs_parm[i]["bandwidth"], bs_parm[i]["numerology"], bs_parm[i]["max_bitrate"], bs_parm[i]["power"], bs_parm[i]["gain"], bs_parm[i]["loss"]))

env.add_base_station(SatelliteBaseStation(env, i+1, (250, 500, 35786000)))

counter = 1000
while counter != 0:
    env.render()
    env.step()
    counter -= 1


