from wns2.basestation.nrbasestation import NRBaseStation
from wns2.userequipment.userequipment import UserEquipment
from wns2.environment.environment import Environment
import numpy.random as random


x_lim = 1000
y_lim = 1000
env = Environment(x_lim, y_lim)

for i in range(0, 10):
    pos = (random.rand()*x_lim, random.rand()*y_lim, 1)
    env.add_user(UserEquipment(env, i, 5, pos, speed = 10, direction = random.randint(0, 360), _lambda_c=5, _lambda_d = 20))

bs_parm =[{"pos": (500, 750, 40),
    "freq": 800,
    "numerology": 1, 
    "power": 20,
    "gain": 16,
    "loss": 3,
    "bandwidth": 20,
    "max_bitrate": 1000},
    
    #BS2
    {"pos": (250, 250, 40),
    "freq": 1700,
    "numerology": 1, 
    "power": 20,
    "gain": 16,
    "loss": 3,
    "bandwidth": 40,
    "max_bitrate": 1000},

    #BS3
    {"pos": (500, 125, 40),
    "freq": 1900,
    "numerology": 1, 
    "power": 20,
    "gain": 16,
    "loss": 3,
    "bandwidth": 40,
    #15
    "max_bitrate": 1000},

    #BS4
    {"pos": (750, 250, 40),
    "freq": 2000,
    "numerology": 1, 
    "power": 20,
    "gain": 16,
    "loss": 3,
    "bandwidth": 25,
    "max_bitrate": 1000}]

for i in range(4): # range(len(bs_parm)):
    env.add_base_station(NRBaseStation(env, i, bs_parm[i]["pos"], bs_parm[i]["freq"], bs_parm[i]["bandwidth"], bs_parm[i]["numerology"], bs_parm[i]["max_bitrate"], bs_parm[i]["power"], bs_parm[i]["gain"], bs_parm[i]["loss"]))

counter = 100
while counter != 0:
    env.render()
    env.step()
    counter -= 1

for bs in env.bs_list:
    print(env.bs_by_id(bs).load_history)

