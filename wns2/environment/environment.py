import logging
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import numpy as np
MIN_RSRP = -140

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)

class Environment:
    def __init__(self, h, l, sampling_time = 1):
        self.h = h
        self.l = l
        self.ue_list = {}
        self.bs_list = {}
        self.sampling_time = sampling_time

        self.plt_run = 0
        return
    
    def add_user(self, ue):
        if ue.get_id() in self.ue_list:
            raise Exception("UE ID mismatch for ID %s", ue.get_id())
        self.ue_list[ue.get_id()] = ue
        return
    
    def remove_user(self, ue_id):
        if ue_id in self.ue_list:
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
            if rsrp_i > MIN_RSRP:
                rsrp[bs] = rsrp_i
        return rsrp
        
    def step(self):
        for ue in self.ue_list:
            self.ue_list[ue].step()
        for bs in self.bs_list:
            self.bs_list[bs].step()
        return

    
    def render(self):
        if self.plt_run == 0:
            plt.ion()
            self.fig = plt.figure(constrained_layout=True)
            self.gs = gridspec.GridSpec(ncols=6, nrows=2*len(self.bs_list), figure=self.fig)
            self.plt_run = 1
        
        x_ue = []
        y_ue = []
        x_bs = []
        y_bs = []

        plt.clf()        
        self.ax = self.fig.add_subplot(self.gs[:,0:2])
        self.axs = {}
        self.axz = {}
        for elem in self.bs_list:
            self.axs[elem] = self.fig.add_subplot(self.gs[elem*2:elem*2+1, 2:4])
        for elem in self.bs_list:
            self.axz[elem] = self.fig.add_subplot(self.gs[elem*2:elem*2+1, 4:6])

        #ax.set_xlim(0, env.x_limit)
        #ax.set_ylim(0, env.y_limit)

        colors = cm.rainbow(np.linspace(0, 1, len(self.bs_list)))

        for j in self.bs_list:
            x_bs.append(self.bs_by_id(j).get_position()[0])
            y_bs.append(self.bs_by_id(j).get_position()[1])

        for i in self.ue_list:
            x_ue.append(self.ue_by_id(i).get_position()[0])
            y_ue.append(self.ue_by_id(i).get_position()[1])

        for i in self.ue_list:
            for j in self.bs_list:
                if j in self.ue_by_id(i).bs_data_rate_allocation:
                    self.ax.scatter(x_ue[i], y_ue[i], color = colors[j])
                    break
            else:
                self.ax.scatter(x_ue[i], y_ue[i], color = "tab:grey")

        for i in self.ue_list:
            self.ax.annotate(str(i+1), (x_ue[i], y_ue[i]))

        for j in self.bs_list:
            if self.bs_by_id(j).bs_type == "drone_relay":
                self.ax.scatter(x_bs[j], y_bs[j], color = colors[j], label = "BS", marker = "^", s = 400, edgecolor = colors[self.bs_by_id(j).linked_bs], linewidth = 3)
            elif self.bs_by_id(j).bs_type == "drone_bs":
                self.ax.scatter(x_bs[j], y_bs[j], color = colors[j], label = "BS", marker = "^", s = 400)
            else:
                self.ax.scatter(x_bs[j], y_bs[j], color = colors[j], label = "BS", marker = "s", s = 400)
        
        for j in self.bs_list:
            self.ax.annotate("BS"+str(j), (x_bs[j], y_bs[j]))
        
        for elem in self.bs_list:
            self.axs[elem].plot(np.arange(0,len(self.bs_list[elem].load_history)), self.bs_list[elem].load_history, color = colors[elem])
            self.axs[elem].grid(True)
            self.axs[elem].set_ylabel("%")
            self.axs[elem].set_xlabel("timestep")
            self.axs[elem].set_ylim(0,1.1)
        for elem in self.bs_list:
            self.axz[elem].plot(np.arange(0,len(self.bs_list[elem].data_rate_history)), self.bs_list[elem].data_rate_history, color = colors[elem])
            self.axz[elem].grid(True)
            self.axz[elem].set_ylabel("Mbps")
            self.axz[elem].set_xlabel("timestep")
            

        self.ax.grid(True)
        self.ax.set_ylabel("[m]")
        self.ax.set_xlabel("[m]")
        self.ax.set_xlim(0, self.l)
        self.ax.set_ylim(0, self.h)
        #plt.draw()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

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
