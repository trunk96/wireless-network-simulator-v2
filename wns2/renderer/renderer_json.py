
import numpy as np
import json
import copy
import os
#import scipy.io as sc


class JSONRendererARIES:
    def __init__(self, path = "data"):
        self.path = path
        if not os.path.isdir(self.path):
            os.mkdir(path)
        self.ue_input_datarate = {}
        self.ue_input_datarate_file = os.path.join(self.path, "input_datarate.json")
        self.ue_desired_bitrate = {}
        self.ue_desired_bitrate_file = os.path.join(self.path, "desired_bitrate.json")
        self.ue_actual_bitrate = {}
        self.ue_actual_bitrate_file = os.path.join(self.path, "actual_bitrate.json")
        self.ue_queue = {}
        self.ue_queue_file = os.path.join(self.path, "queue.json")
        self.ue_queue_out = {}
        self.ue_queue_out_file = os.path.join(self.path, "queue_out.json")
        self.bs_bitrate = {}
        self.bs_bitrate_file = os.path.join(self.path, "bs_bitrate.json")
        self.bs_load = {}
        self.bs_load_file = os.path.join(self.path, "bs_load.json")
        self.drone_pos = {}
        self.drone_pos_file = os.path.join(self.path, "drone_pos.json")

        self.counter = 0

        return

    def render(self, env):
        N = len(env.ue_list)
        M = len(env.bs_list)
        self.ue_input_datarate[self.counter] = {}
        self.ue_desired_bitrate[self.counter] = {}
        self.ue_actual_bitrate[self.counter] = {}
        self.ue_queue[self.counter] = {}
        self.ue_queue_out[self.counter] = {}
        self.bs_bitrate[self.counter] = {}
        self.bs_load[self.counter] = {}
        self.drone_pos[self.counter] = {}
        for ue in env.ue_list:
            ue = env.ue_by_id(ue)
            ueid = ue.get_id()
            self.ue_input_datarate[self.counter][ueid] = ue.get_current_input_data_rate()
            self.ue_desired_bitrate[self.counter][ueid] = copy.deepcopy(ue.output_data_rate)
            self.ue_actual_bitrate[self.counter][ueid] = copy.deepcopy(ue.bs_data_rate_allocation)
            self.ue_queue[self.counter][ueid] = ue.queue
            if ue.queue_out:
                self.ue_queue_out[self.counter][ueid] = 0
            else:
                self.ue_queue_out[self.counter][ueid] = 1
        for bs in env.bs_list:
            bs = env.bs_by_id(bs)
            bsid = bs.get_id()
            self.bs_bitrate[self.counter][bsid] = bs.get_allocated_data_rate()
            self.bs_load[self.counter][bsid] = bs.get_usage_ratio()
            if bs.bs_type == "drone":
                self.drone_pos[self.counter][bsid] = bs.position
        self.counter += 1
        with open(self.ue_input_datarate_file, "w") as fp:
            json.dump(self.ue_input_datarate, fp)
        with open(self.ue_desired_bitrate_file, "w") as fp:
            json.dump(self.ue_desired_bitrate, fp)
        with open(self.ue_actual_bitrate_file, "w") as fp:
            json.dump(self.ue_actual_bitrate, fp)
        with open(self.ue_queue_file, "w") as fp:
            json.dump(self.ue_queue, fp)
        with open(self.ue_queue_out_file, "w") as fp:
            json.dump(self.ue_queue_out, fp)
        with open(self.bs_bitrate_file, "w") as fp:
            json.dump(self.bs_bitrate, fp)
        with open(self.bs_load_file, "w") as fp:
            json.dump(self.bs_load, fp)
        with open(self.drone_pos_file, "w") as fp:
            json.dump(self.drone_pos, fp)
        return
