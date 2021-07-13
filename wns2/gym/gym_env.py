import gym
from gym import spaces
from wns2.basestation.nrbasestation import NRBaseStation
from wns2.basestation.satellitebasestation import SatelliteBaseStation
from wns2.userequipment.userequipment import UserEquipment
from wns2.environment.environment import Environment
from wns2.renderer.renderer import CustomRenderer
import numpy.random as random
import logging
import numpy as np


class WNSEnv(gym.Env):
    metadata = {'render.modes': ['human']}

def __init__(self, x_lim, y_lim, n_ue, terr_parm, sat_parm):
    super(WNSEnv, self).__init__()
    # --------terr_parm example:------------
    # {"pos": (500, 500, 30),
    # "freq": 800,
    # "numerology": 1, 
    # "power": 20,
    # "gain": 16,
    # "loss": 3,
    # "bandwidth": 20,
    # "max_bitrate": 1000}
    # 
    # --------sat_parm example:------------
    # {"pos": (250, 500, 35786000)}
    len_p = len(terr_parm)+len(sat_parm)
    # Define action and observation space
    # They must be gym.spaces objects
    # Example when using discrete actions:
    self.action_space = spaces.Discrete(len_p+1)
    # Example for using image as input:
    self.observation_space = spaces.Box(low=0, high=1, shape=(2*len_p+1, 1))

    env = Environment(x_lim, y_lim, renderer = CustomRenderer())
    for i in range(0, n_ue):
        pos = (random.rand()*x_lim, random.rand()*y_lim, 1)
        env.add_user(UserEquipment(env, i, 25, pos, speed = 10, direction = random.randint(0, 360), _lambda_c=5, _lambda_d = 15))
    for i in range(len(terr_parm)):
        env.add_base_station(NRBaseStation(env, i, terr_parm[i]["pos"], terr_parm[i]["freq"], terr_parm[i]["bandwidth"], terr_parm[i]["numerology"], terr_parm[i]["max_bitrate"], terr_parm[i]["power"], terr_parm[i]["gain"], terr_parm[i]["loss"]))
    for i in range(len(sat_parm)):
        env.add_base_station(SatelliteBaseStation(env, len(terr_parm)+i, sat_parm["pos"]))

    def step(self, action):
        return observation, reward, done, info
    def reset(self):
        return observation  # reward, done, info can't be included
    def render(self, mode='human'):
        return
    def close (self):
        return
 
