import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import numpy as np


class CustomRenderer:
    def __init__(self):
        self.plt_run = 0
        return

    def render(self, env):
        if self.plt_run == 0:
            plt.ion()
            self.fig = plt.figure(constrained_layout=True)
            self.gs = gridspec.GridSpec(ncols=6, nrows=2*len(env.bs_list), figure=self.fig)
            self.plt_run = 1
        
        x_ue = []
        y_ue = []
        x_bs = []
        y_bs = []

        plt.clf()        
        self.ax = self.fig.add_subplot(self.gs[:,0:2])
        self.axs = {}
        self.axz = {}
        for elem in env.bs_list:
            self.axs[elem] = self.fig.add_subplot(self.gs[elem*2:elem*2+1, 2:4])
        for elem in env.bs_list:
            self.axz[elem] = self.fig.add_subplot(self.gs[elem*2:elem*2+1, 4:6])

        #ax.set_xlim(0, env.x_limit)
        #ax.set_ylim(0, env.y_limit)

        colors = cm.rainbow(np.linspace(0, 1, len(env.bs_list)))

        for j in env.bs_list:
            x_bs.append(env.bs_by_id(j).get_position()[0])
            y_bs.append(env.bs_by_id(j).get_position()[1])

        for i in env.ue_list:
            x_ue.append(env.ue_by_id(i).get_position()[0])
            y_ue.append(env.ue_by_id(i).get_position()[1])

        for i in env.ue_list:
            for j in env.bs_list:
                if j in env.ue_by_id(i).bs_data_rate_allocation:
                    self.ax.scatter(x_ue[i], y_ue[i], color = colors[j])
                    break
            else:
                self.ax.scatter(x_ue[i], y_ue[i], color = "tab:grey")

        for i in env.ue_list:
            self.ax.annotate(str(i+1), (x_ue[i], y_ue[i]))

        for j in env.bs_list:
            if env.bs_by_id(j).bs_type == "drone_relay":
                self.ax.scatter(x_bs[j], y_bs[j], color = colors[j], label = "BS", marker = "^", s = 400, edgecolor = colors[env.bs_by_id(j).linked_bs], linewidth = 3)
            elif env.bs_by_id(j).bs_type == "drone_bs":
                self.ax.scatter(x_bs[j], y_bs[j], color = colors[j], label = "BS", marker = "^", s = 400)
            else:
                self.ax.scatter(x_bs[j], y_bs[j], color = colors[j], label = "BS", marker = "s", s = 400)
        
        for j in env.bs_list:
            self.ax.annotate("BS"+str(j), (x_bs[j], y_bs[j]))
        
        for elem in env.bs_list:
            self.axs[elem].plot(np.arange(0,len(env.bs_list[elem].load_history)), env.bs_list[elem].load_history, color = colors[elem])
            self.axs[elem].grid(True)
            self.axs[elem].set_ylabel("%")
            self.axs[elem].set_xlabel("timestep")
            self.axs[elem].set_ylim(0,1.1)
        for elem in env.bs_list:
            self.axz[elem].plot(np.arange(0,len(env.bs_list[elem].data_rate_history)), env.bs_list[elem].data_rate_history, color = colors[elem])
            self.axz[elem].grid(True)
            self.axz[elem].set_ylabel("Mbps")
            self.axz[elem].set_xlabel("timestep")
            

        self.ax.grid(True)
        self.ax.set_ylabel("[m]")
        self.ax.set_xlabel("[m]")
        self.ax.set_xlim(0, env.l)
        self.ax.set_ylim(0, env.h)
        #plt.draw()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
