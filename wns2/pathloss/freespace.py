from wns2.pathloss.generic import GenericPathLoss
import math
import scipy.constants
import logging

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)

class FreeSpacePathLoss(GenericPathLoss):
    def __init__(self):
        return
    
    def compute_path_loss(self, ue, bs):
        ue_pos = ue.get_position()
        bs_pos = bs.get_position()
        dist = math.sqrt((ue_pos[0]-bs_pos[0])**2 + (ue_pos[1]-bs_pos[1])**2 + (ue_pos[2] - bs_pos[2])**2)
        bs_frequency = bs.get_carrier_frequency()*1e9 #from GHz to Hz
        
        path_loss = 10*math.log10(((4*scipy.constants.pi*dist*bs_frequency)/scipy.constants.speed_of_light)**2)
        return path_loss 