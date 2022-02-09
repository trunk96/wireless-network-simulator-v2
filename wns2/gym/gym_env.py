import gym
from gym import spaces
from wns2.basestation.nrbasestation import NRBaseStation
from wns2.basestation.satellitebasestation import SatelliteBaseStation
from wns2.userequipment.multipath_userequipment import MultiPathUserEquipment
from wns2.environment.environment import Environment
from wns2.renderer.renderer import CustomRenderer
import numpy.random as random
import logging
import numpy as np
import math


logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)

class WNSEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def init_env(self, x_lim, y_lim, terr_parm, sat_parm, n_ue):
        self.env = Environment(x_lim, y_lim, renderer = CustomRenderer())
        self.init_pos = []  # for reset method
        for i in range(0, n_ue):
            pos = (random.rand()*x_lim, random.rand()*y_lim, 1)
            self.env.add_user(MultiPathUserEquipment(self.env, i, 25, pos, speed = 0, direction = random.randint(0, 360), _lambda_c=5, _lambda_d = 15))
            self.init_pos.append(pos)
        for i in range(len(terr_parm)):
            self.env.add_base_station(NRBaseStation(self.env, i, terr_parm[i]["pos"], terr_parm[i]["freq"], terr_parm[i]["bandwidth"], terr_parm[i]["numerology"], terr_parm[i]["max_bitrate"], terr_parm[i]["power"], terr_parm[i]["gain"], terr_parm[i]["loss"]))
        for i in range(len(sat_parm)):
            self.env.add_base_station(SatelliteBaseStation(self.env, len(terr_parm)+i, sat_parm[i]["pos"]))

    def __init__(self, x_lim, y_lim, n_ue, terr_parm, sat_parm, queue_parm, load_parm, max_steps):
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
        # --------queue_parm example:------------
        # [q0, q1]
        # --------queue_parm example:------------
        # [l0, l1, l2]
        self.len_p = len(terr_parm)+len(sat_parm)
        self.n_ue = n_ue
        self.queue_parm = queue_parm
        self.load_parm = load_parm
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.MultiDiscrete([2**self.len_p for _ in range(self.n_ue)]) 
        # Example for using image as input:
        self.observation_space = spaces.MultiDiscrete([3 for _ in range(self.n_ue)]+[4 for _ in range(self.len_p)])
        self.steps = max_steps
        self.max_steps = max_steps
        self.terr_param = terr_parm
        self.sat_param = sat_parm

        self.init_env(x_lim, y_lim, terr_parm, sat_parm, n_ue)

    def observe(self):
        ue_obs = []
        for i in range(self.n_ue):
            q = self.env.ue_by_id(i).get_queue()
            if q < self.queue_parm[0]:
                ue_obs.append(0)
            elif q < self.queue_parm[1]:
                ue_obs.append(1)
            else:
                ue_obs.append(2)
        
        bs_obs = []
        for j in range(self.len_p):
            l = self.env.bs_by_id(j).get_usage_ratio()
            if l < self.load_parm[0]:
                bs_obs.append(0)
            elif l < self.load_parm[1]:
                bs_obs.append(1)
            elif l < self.load_parm[2]:
                bs_obs.append(2)
            else:
                bs_obs.append(3)
        observation = np.array(ue_obs+bs_obs)
        return observation

    
    def step(self, action):
        # action represents which APs the UEs have to connect to
        
        connection_matrix = []
        for i in range(self.n_ue):
            connection_matrix.append([0] * self.len_p)
            for j in range(self.len_p):
                if (action[i] >> j) & 1 == 1:
                    connection_matrix[i][j] = 1
        for i in range(self.n_ue):
            ue = self.env.ue_by_id(i)
            if ue.get_current_input_data_rate() == 0:
                continue
            connection_list = []
            for bs in range(self.len_p):
                if connection_matrix[i][bs] == 1:
                    connection_list.append(bs)
            if len(connection_list) == 0:
                continue
            for bs in range(self.len_p):
                if connection_matrix[i][bs] == 1:
                    ue.output_data_rate[bs] = ue.get_current_input_data_rate()/len(connection_list)
            ue.connect_bs(connection_list)
        
        self.env.step()
        

        done = False
        if self.steps <= 0:
            done = True
        self.steps -= 1

        info = None

        observation = self.observe()            
        
        reward = 0.0
        lambda_q = 1.0
        lambda_l = 1.0
        delta_q = self.n_ue*lambda_q
        delta_l = self.len_p*lambda_l
        
        ue_obs = observation[0:self.n_ue]
        bs_obs = observation[self.n_ue:]
        for i in range(self.n_ue):
            reward -= lambda_q*math.exp(ue_obs[i])
        reward += delta_q
        for j in range(self.len_p):
            reward -= lambda_l*math.exp(bs_obs[j])
        reward += delta_l

        for i in range(self.n_ue):
            self.env.ue_by_id(i).disconnect_all()

        return observation, reward, done, info

    def reset(self):
        x_lim = self.env.get_x_limit()
        y_lim = self.env.get_y_limit()

        self.init_env(x_lim, y_lim, self.terr_param, self.sat_param, self.n_ue)
        observation = self.observe()
        return observation  # reward, done, info can't be included

    def render(self, mode='human'):
        return self.env.render()
    def close (self):
        return
 
