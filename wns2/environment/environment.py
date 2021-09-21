import logging
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import numpy as np
import wns2.environment.util as util
MIN_RSRP = -140

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)

class Environment:
    def __init__(self, h, l, sampling_time = 1, renderer = None):
        self.h = h
        self.l = l
        self.ue_list = {}
        self.connection_advertisement = []
        self.bs_list = {}
        self.sampling_time = sampling_time # in seconds
        self.renderer = renderer
        self.plt_run = 0
        return
    
    def add_user(self, ue):
        if ue.get_id() in self.ue_list:
            raise Exception("UE ID mismatch for ID %s", ue.get_id())
        self.ue_list[ue.get_id()] = ue
        return
    
    def remove_user(self, ue_id):
        if ue_id in self.ue_list:
            if self.ue_list[ue_id].get_current_bs() != None:
                bs = self.ue_list[ue_id].get_current_bs()
                self.ue_list[ue_id].disconnect(bs)
            del self.ue_list[ue_id]

    def add_base_station(self, bs):
        if bs.get_id() in self.bs_list:
            raise Exception("BS ID mismatch for ID %s", bs.get_id())
        self.bs_list[bs.get_id()] = bs
        return

    def compute_rsrp(self, ue):
        rsrp = {}
        for bs in self.bs_list:
            rsrp_i = self.bs_list[bs].compute_rsrp(ue)
            if rsrp_i > MIN_RSRP or self.bs_list[bs].get_bs_type() == "sat":
                rsrp[bs] = rsrp_i
        return rsrp
    
    def advertise_connection(self, ue_id):
        self.connection_advertisement.append(ue_id)
        return
        
    def step(self): 
        self.connection_advertisement.clear()
        for ue in self.ue_list:
            self.ue_list[ue].step()
        for bs in self.bs_list:
            self.bs_list[bs].step()
               
        # call here the optimizator, taking the w from users (with data_generation_status and input_data_rate)
        # set the desired data rate for all the users, connecting to the BSs
        # move the drone AP according to the weighted average of the positions of the UEs connected to
        for i in range(len(self.ue_list)):
            self.ue_by_id(i).disconnect_all()
        N = len(self.connection_advertisement)
        M = len(self.bs_list)
        q = np.zeros(N)
        w = np.zeros(N)
        P = np.zeros((N, M))
        for i in range(N):
            ue = self.ue_by_id(self.connection_advertisement[i])
            q[i] = ue.get_queue()
            w[i] = ue.get_current_input_data_rate()
            rsrp = ue.measure_rsrp()
            for bs in rsrp:
                P[i, bs] = 1
        u_final = util.output_datarate_optimization(q, w, N, M, P, self.sampling_time)
        for i in range(N):
            ue = self.ue_by_id(self.connection_advertisement[i])
            connection_list = []
            for bs in range(M):
                if u_final[i, bs] > 0 and P[i, bs] == 1:
                    ue.output_data_rate[bs] = u_final[i, bs]
                    connection_list.append(bs)
            ue.connect_bs(connection_list)
        return            

    
    def render(self):
        if self.renderer != None:
            return self.renderer.render(self)

    def bs_by_id(self, id):
        return self.bs_list[id]
    def ue_by_id(self, id):
        return self.ue_list[id]
    def get_sampling_time(self):
        return self.sampling_time
    def get_x_limit(self):
        return self.l
    def get_y_limit(self):
        return self.h
