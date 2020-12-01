from wns2.pathloss.generic import GenericPathLoss
import math
from enum import Enum
import logging

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)

class EnvType(Enum):
    RURAL = 0
    SUBURBAN = 1
    URBAN = 2

class CostHataPathLoss(GenericPathLoss):
    def __init__(self, env_type = EnvType.URBAN):
        self.env_type = env_type
        return
    
    def compute_path_loss(self, ue, bs):
        #check feasibility
        ue_pos = ue.get_position()
        bs_pos = bs.get_position()
        bs_frequency = bs.get_carrier_frequency()
        if bs_frequency > 2000 or bs_frequency < 150:
            logging.warning("Cost Hata model is designed for carrier frequency in [150, 2000] MHz, check the results obtained before using them")    
        if ue_pos[2] > 10 or ue_pos[2] < 1:
            logging.warning("Cost Hata model is designed for UE height in [1, 10] m, check the results obtained before using them")
        if bs_pos[2] > 200 or bs_pos[2] < 30:
            logging.warning("Cost Hata model is designed for BS height in [30, 200] m, check the results obtained before using them")
        
        #compute distance first
        dist = math.sqrt((ue_pos[0]-bs_pos[0])**2 + (ue_pos[1]-bs_pos[1])**2 + (ue_pos[2] - bs_pos[2])**2)
        if dist == 0:   #just to avoid log(0) in path loss computing
            dist = 0.01
        #compute C_0, C_f, b(h_b), a(h_m) and C_m with the magic numbers defined by the model
        if bs_frequency <= 1500 and bs_frequency >= 150 :
            C_0 = 69.55
            C_f = 26.16
            b = 13.82*math.log10(bs_pos[2])
            if self.env_type == EnvType.URBAN:
                C_m = 0
            elif self.env_type == EnvType.SUBURBAN:
                C_m = -2*((math.log10(bs_frequency/28))**2) - 5.4
            else:
                C_m = -4.78*((math.log10(bs_frequency))**2) + 18.33*math.log10(bs_frequency) - 40.94
        else:  
            C_0 = 46.3
            C_f = 26.16
            b = 13.82*math.log10(bs_pos[2])
            if self.env_type == EnvType.URBAN:
                C_m = 3
            elif self.env_type == EnvType.SUBURBAN:
                C_m = 0
            else:
                raise Exception("COST-HATA model is not defined for frequencies in 1500-2000MHz with RURAL environments")
        
        if self.env_type == EnvType.SUBURBAN or self.env_type == EnvType.RURAL:
            a = (1.1*math.log10(bs_frequency) - 0.7)*ue_pos[2] - 1.56*math.log10(bs_frequency) + 0.8
        else:
            if bs_frequency >= 150 and bs_frequency <= 300:
                a = 8.29*(math.log10(1.54*ue_pos[2])**2) - 1.1
            else:
                a = 3.2*(math.log10(11.75*ue_pos[2])**2) - 4.97
        
        path_loss = C_0 + C_f * math.log10(bs_frequency) - b - a + (44.9-6.55*math.log10(bs_pos[2]))*math.log10(dist/1000) + C_m
        return path_loss
        
