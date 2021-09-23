from wns2.basestation.generic import BaseStation
import math
from scipy import constants
from wns2.environment import environment
from wns2.pathloss import costhata
import logging

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)

MAX_PRB = 200

#Table 5.3.3-1: Minimum guardband [kHz] (FR1) and Table: 5.3.3-2: Minimum guardband [kHz] (FR2), 3GPPP 38.104
#number of prb depending on the numerology (0,1,2,3), on the frequency range (FR1, FR2) and on the base station bandwidth
NRbandwidth_prb_lookup = {
    0:[{
        5:25,
        10:52,
        15:79,
        20:106,
        25:133,
        30:160,
        40:216,
        50:270
    }, None],
    1:[{
        5:11,
        10:24,
        15:38,
        20:51,
        25:65,
        30:78,
        40:106,
        50:133,
        60:162,
        70:189,
        80:217,
        90:245,
        100:273
    }, None],
    2:[{
        10:11,
        15:18,
        20:24,
        25:31,
        30:38,
        40:51,
        50:65,
        60:79,
        70:93,
        80:107,
        90:121,
        100:135
    },
    {
        50:66,
        100:132,
        200:264
    }],
    3:[None, 
    {
        50:32,
        100:66,
        200:132,
        400:264
    }]
}

class NRBaseStation(BaseStation):
    def __init__(self, env, bs_id, position, carrier_frequency, total_bandwidth, numerology, max_data_rate = None, antenna_power = 20, antenna_gain = 16, feeder_loss = 3, pathloss = None):
        if numerology not in NRbandwidth_prb_lookup:
            raise Exception("Invalid numerology for Base Station "+str(bs_id))
        if carrier_frequency >= 410 and carrier_frequency <=7125: # MHz
            self.fr = 0 # Frequency Range 1 (sub 6GHz)
        elif carrier_frequency >= 24250 and carrier_frequency <= 52600: #MHz
            self.fr = 1 # Frequency Range 2 (24.25-52.6GHz)
        else:
            raise Exception("Invalid carirer frequency for Base Station "+str(bs_id))
        if total_bandwidth not in NRbandwidth_prb_lookup[numerology][self.fr]:
            raise Exception("Invalid total bandwith for Base Station "+str(bs_id))
        self.env = env
        self.bs_id = bs_id
        self.bs_type = "nr"
        self.position = position
        self.carrier_frequency = carrier_frequency
        self.total_bandwidth = total_bandwidth
        self.numerology = numerology
        self.antenna_power = antenna_power
        self.antenna_gain = antenna_gain
        self.feeder_loss = feeder_loss
        if pathloss == None:
            self.pathloss = costhata.CostHataPathLoss(costhata.EnvType.URBAN)
        else:
            self.pathloss = pathloss
        self.max_data_rate = max_data_rate

        self.total_prb = NRbandwidth_prb_lookup[self.numerology][self.fr][self.total_bandwidth] * (10 * 2**self.numerology) # 10*2^mu time slots in a time frame
        self.subcarrier_bandwidth = 15*(2**self.numerology) #KHz

        # allocation structures
        self.ue_pb_allocation = {}
        self.ue_data_rate_allocation = {}
        self.allocated_prb = 0
        self.allocated_data_rate = 0
        self.T = 10
        self.resource_utilization_array = [0] * self.T
        self.resource_utilization_counter = 0
        self.load_history = []
        self.data_rate_history = []
        return

    def get_position(self):
        return self.position
    def get_carrier_frequency(self):
        return self.carrier_frequency
    def get_bs_type(self):
        return self.bs_type
    def get_id(self):
        return self.bs_id
    def get_usage_ratio(self):
        return self.allocated_prb / self.total_prb

    def compute_rsrp(self, ue):
        subcarrier_power = 10*math.log10(self.antenna_power*1000 / (12*(self.total_prb/(10*2**self.numerology))))
        return subcarrier_power + self.antenna_gain -self.feeder_loss - self.pathloss.compute_path_loss(ue, self)

    def get_rbur(self):
        return sum(self.resource_utilization_array)/(self.T*self.total_prb)

    def compute_sinr(self, rsrp):
        interference = 0
        for elem in rsrp:
            bs_i = self.env.bs_by_id(elem)
            if elem != self.bs_id and bs_i.get_carrier_frequency() == self.carrier_frequency:
                rbur_i = bs_i.get_rbur()
                interference += (10 ** (rsrp[elem]/10))*rbur_i
        thermal_noise = constants.Boltzmann*293.15*15*(2**self.numerology)*1000 # delta_F = 15*2^mu KHz each subcarrier since we are considering measurements at subcarrirer level (like RSRP)
        sinr = (10**(rsrp[self.bs_id]/10))/(thermal_noise + interference)
        logging.debug("BS %s -> SINR: %s", self.bs_id, str(10*math.log10(sinr)))
        return sinr
    
    def compute_prb_NR(self, data_rate, rsrp):
        sinr = self.compute_sinr(rsrp)
        r = 12*self.subcarrier_bandwidth*1e3*math.log2(1+sinr)*(1/(10*(2**self.numerology))) # if a single RB is allocated we transmit for 1/(10*2^mu) seconds each second in 12*15*2^mu KHz bandwidth
        n_prb = math.ceil(data_rate*1e6/r) # the data-rate is in Mbps, so we had to convert it
        return n_prb, r/1e6

    def connect(self, ue_id, desired_data_rate, rsrp):
        # compute the number of PRBs needed for the requested data-rate,
        # then allocate them as much as possible

        n_prb, r = self.compute_prb_NR(desired_data_rate, rsrp)

        if self.max_data_rate != None:
            if self.max_data_rate - self.allocated_data_rate < r*n_prb:
                data_rate = self.max_data_rate - self.allocated_data_rate
                if data_rate < 0:
                    data_rate = 0 # due to computational errors
                n_prb, r = self.compute_prb_NR(data_rate, rsrp)

        if self.total_prb - self.allocated_prb < n_prb:
            n_prb = self.total_prb - self.allocated_prb
        
        if MAX_PRB != -1 and n_prb > MAX_PRB and self.get_usage_ratio() > 0.8:
            n_prb = MAX_PRB
        
        if ue_id in self.ue_pb_allocation:
            self.allocated_prb -= self.ue_pb_allocation[ue_id]
        self.ue_pb_allocation[ue_id] = n_prb
        self.allocated_prb += n_prb 

        if ue_id in self.ue_data_rate_allocation:
            self.allocated_data_rate -= self.ue_data_rate_allocation[ue_id]
        self.ue_data_rate_allocation[ue_id] = n_prb*r
        self.allocated_data_rate += n_prb*r 
        return r*n_prb
    
    def disconnect(self, ue_id):
        self.allocated_prb -= self.ue_pb_allocation[ue_id]
        self.allocated_data_rate -= self.ue_data_rate_allocation[ue_id]
        del self.ue_data_rate_allocation[ue_id]
        del self.ue_pb_allocation[ue_id]
        return
    
    def update_connection(self, ue_id, desired_data_rate, rsrp):
        # this can be called if desired_data_rate is changed or if the rsrp is changed
        # compute the number of PRBs needed for the requested data-rate,
        # then allocate them as much as possible
        #self.allocated_prb -= self.ue_pb_allocation[ue_id]
        #self.allocated_data_rate -= self.ue_data_rate_allocation[ue_id]
        self.disconnect(ue_id)
        return self.connect(ue_id, desired_data_rate, rsrp)

    def step(self):
        self.resource_utilization_array[self.resource_utilization_counter] = self.allocated_prb
        self.resource_utilization_counter += 1
        if self.resource_utilization_counter % self.T == 0:
            self.resource_utilization_counter = 0
        
        self.load_history.append(self.get_usage_ratio())
        self.data_rate_history.append(self.allocated_data_rate)
    
    def get_allocated_data_rate(self):
        return self.allocated_data_rate
