from . import generic
from generic import GenericPathLoss
import math
import scipy.constants

class FreeSpacePathLoss(GenericPathLoss):
    def __init__(self):
        super.__init__()
        return
    
    def compute_path_loss(self, ue, bs):
        ue_pos = ue.get_position()
        bs_pos = bs.get_position()
        dist = math.sqrt((ue_pos[0]-bs_pos[0])**2 + (ue_pos[1]-bs_pos[1])**2 + (ue_pos[2] - bs_pos[2])**2)
        bs_frequency = bs.get_carrier_frequency()
        
        path_loss = ((4*scipy.constants.pi*dist*bs_frequency)/scipy.constants.speed_of_light)**2
        return path_loss