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


logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)

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
        self.len_p = len(terr_parm)+len(sat_parm)
        self.n_ue = n_ue
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(self.len_p+1) 
        # Example for using image as input:
        self.observation_space = spaces.Box(low=0, high=1, shape=(2*self.len_p+1, 1))

        self.env = Environment(x_lim, y_lim, renderer = CustomRenderer())
        self.init_pos = []  # for reset method
        for i in range(0, n_ue):
            pos = (random.rand()*x_lim, random.rand()*y_lim, 1)
            self.env.add_user(UserEquipment(self.env, i, 25, pos, speed = 10, direction = random.randint(0, 360), _lambda_c=5, _lambda_d = 15))
            self.init_pos.append(pos)
        for i in range(len(terr_parm)):
            self.env.add_base_station(NRBaseStation(self.env, i, terr_parm[i]["pos"], terr_parm[i]["freq"], terr_parm[i]["bandwidth"], terr_parm[i]["numerology"], terr_parm[i]["max_bitrate"], terr_parm[i]["power"], terr_parm[i]["gain"], terr_parm[i]["loss"]))
        for i in range(len(sat_parm)):
            self.env.add_base_station(SatelliteBaseStation(self.env, len(terr_parm)+i, sat_parm[i]["pos"]))

    def step(self, action):
        # action represents the BS ID to connect to
        # if action is equal to n_BS+1, then it means to not allocate the user to any BS
        logging.warning("Connection Advertisement: "+str(self.env.connection_advertisement))
        if action < self.len_p:
            if 0 in self.env.connection_advertisement:
                logging.warning("Action for UE 0: %s" %action)
                self.env.ue_by_id(0).connect_bs(action)
        for ue in range(1, self.n_ue):
            if ue in self.env.connection_advertisement:
                self.env.ue_by_id(ue).connect_max_rsrp()
        done = False
        info = None
        observation = None  #TODO
        reward = 0          #TODO
        self.env.step()                    
        return observation, reward, done, info
    def reset(self):
        observation = None  #TODO
        return observation  # reward, done, info can't be included
    def render(self, mode='human'):
        return
    def close (self):
        return
 
