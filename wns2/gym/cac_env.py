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
import copy


logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)

class CACGymEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    QUANTIZATION = 5 #0%, 20%, 40%, 60%, 80% 100%
    CLASSES_OF_SERVICE = 3

    def init_env(self, x_lim, y_lim, terr_parm, sat_parm, n_ue):
        self.env = Environment(x_lim, y_lim, renderer = CustomRenderer())
        self.init_pos = []  # for reset method
        for i in range(0, n_ue):
            pos = (random.rand()*x_lim, random.rand()*y_lim, 1)
            self.env.add_user(UserEquipment(self.env, i, 25, pos, speed = 0, direction = random.randint(0, 360), _lambda_c=5, _lambda_d = 15))
            self.init_pos.append(pos)
        for i in range(len(terr_parm)):
            self.env.add_base_station(NRBaseStation(self.env, i, terr_parm[i]["pos"], terr_parm[i]["freq"], terr_parm[i]["bandwidth"], terr_parm[i]["numerology"], terr_parm[i]["max_bitrate"], terr_parm[i]["power"], terr_parm[i]["gain"], terr_parm[i]["loss"]))
        for i in range(len(sat_parm)):
            self.env.add_base_station(SatelliteBaseStation(self.env, len(terr_parm)+i, sat_parm[i]["pos"]))
        self.terr_parm = terr_parm
        self.sat_parm = sat_parm

    def __init__(self, x_lim, y_lim, class_list, terr_parm, sat_parm):
            super(CACGymEnv, self).__init__()
            self.n_ap = len(terr_parm)+len(sat_parm)
            self.action_space = spaces.Discrete(self.n_ap+1)
            #self.observation_space = spaces.MultiDiscrete([self.CLASSES_OF_SERVICE]+[self.QUANTIZATION+1 for _ in range(self.n_ap)])
            self.observation_space = spaces.Discrete(self.CLASSES_OF_SERVICE*((self.QUANTIZATION+1)**self.n_ap))
            self.n_ue = len(class_list)
            self.x_lim = x_lim
            self.y_lim = y_lim
            self.class_list = class_list
            class_set = set(class_list)
            self.number_of_classes = len(class_set)
            self.init_env(x_lim, y_lim, terr_parm, sat_parm, self.n_ue)
    
    def observe(self, ue_id):
        bs_obs = []
        for j in range(self.n_ap):
            l = self.env.bs_by_id(j).get_usage_ratio()
            counter = 0
            for i in np.arange(0, 1, 1/self.QUANTIZATION):
                if l <= i:
                    bs_obs.append(counter)
                    break
                counter += 1
        observation_arr = np.array([self.class_list[ue_id]]+bs_obs)
        observation = 0
        counter = 0
        for elem in observation_arr:
            if counter == 0:
                observation = elem * self.CLASSES_OF_SERVICE
            else:
                observation += elem * ((self.QUANTIZATION+1)**counter)
            counter += 1
        return observation

    def step(self, action):
        # actuate the action on self.current_ue_id
        current_data_rate = None
        if action != 0:
            selected_bs = action - 1
            self.env.ue_by_id(self.current_ue_id).disconnect()
            current_data_rate = self.env.ue_by_id(self.current_ue_id).connect_bs(selected_bs)
        
        # disconnect all UEs that are not wanting to connect
        for ue_id in range(self.n_ue):
            if ue_id not in self.env.connection_advertisement:
                self.env.ue_by_id(ue_id).disconnect()

        done = False
        # compute reward for all the Q tables
        info = np.zeros(self.number_of_classes)
        for i in range(self.number_of_classes):
            if i == self.class_list[self.current_ue_id]:
                if (current_data_rate == None) or (current_data_rate < self.env.ue_by_id(ue_id).data_rate):
                    info[i] = 1
                else:
                    info[i] = 0
            else:
                info[i] = 0
        reward = np.sum(info)
        
        # make the env go 1 step forward
        self.env.step()

        # select next ue that will be scheduled (if all the UEs are scheduled yet, fast-forward steps in the environment)
        if len(self.advertised_connections) > 0:
            for ue_id in range(self.n_ue):
                self.env.ue_by_id(ue_id).last_time -= 1
        else:
            self.env.step()
            self.advertised_connections = copy.deepcopy(self.env.connection_advertisement)
            while len(self.advertised_connections) == 0:
                self.env.step()
                self.advertised_connections = copy.deepcopy(self.env.connection_advertisement)
        
        next_ue_id = random.choice(self.advertised_connections)
        self.advertised_connections.remove(next_ue_id)
        next_ue = self.env.ue_by_id(next_ue_id)
        # if next_ue is already connected to an AP, skip it and focus only on the unconnected UEs
        while next_ue.get_current_bs() != None:
            while len(self.advertised_connections) == 0:
                self.env.step()
                self.advertised_connections = copy.deepcopy(self.env.connection_advertisement)
            next_ue_id = random.choice(self.advertised_connections)
            self.advertised_connections.remove(next_ue_id)
            next_ue = self.env.ue_by_id(next_ue_id)

        
        self.current_ue_id = next_ue_id
        # after the step(), the user that have to appear in the next state is the next user, not the current user
        observation = self.observe(next_ue_id)
                
        return observation, reward, done, info

    def reset(self):
        self.init_env(self.x_lim, self.y_lim, self.terr_parm, self.sat_parm, self.n_ue)
        self.env.step()
        self.advertised_connections = copy.deepcopy(self.env.connection_advertisement)
        # step until at least one UE wants to connect
        while len(self.advertised_connections) == 0:
            self.env.step()
            self.advertised_connections = copy.deepcopy(self.env.connection_advertisement)
        ue_id = random.choice(self.advertised_connections)
        self.current_ue_id = ue_id
        # go back 1 time instant, so at the next step() the connection_advertisement list will not change
        for _ue_id in range(self.n_ue):
            self.env.ue_by_id(_ue_id).last_time -= 1
        observation = self.observe(ue_id)
        self.advertised_connections.remove(ue_id)
        return observation

    def render(self, mode='human'):
        return self.env.render()
    
    def close (self):
        return