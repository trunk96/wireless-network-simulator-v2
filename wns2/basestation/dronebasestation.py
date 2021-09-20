from wns2.basestation.nrbasestation import NRBaseStation
import math
from scipy import constants
from wns2.environment import environment
from wns2.pathloss import costhata
import logging

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)


class DroneBaseStation(NRBaseStation):   
    def __init__(self, env, bs_id, position, carrier_frequency, total_bandwidth, numerology, max_data_rate = None, antenna_power = 20, antenna_gain = 16, feeder_loss = 3, pathloss = None):
        self.theta_k = 0
        self.bs_type = "drone"
        return super().__init__(env, bs_id, position, carrier_frequency, total_bandwidth, numerology, max_data_rate = None, antenna_power = 20, antenna_gain = 16, feeder_loss = 3, pathloss = None)

    def move(self, destination, speed):
        logging.debug("Destination %s %s %s" %(destination[0], destination[1], destination[2]))
        speed *= self.env.get_sampling_time() # scale the speed (that is given in m/s) according to the environment sampling time
        x_k = destination[0] - self.position[0]
        y_k = destination[1] - self.position[1]
        z_k = destination[2] - self.position[2]
        theta_k = self.theta_k
        v_k = 1*(x_k*math.cos(theta_k) + y_k*math.sin(theta_k))
        v_z_k = 1*z_k
        if v_k > speed and v_k > 0:
            v_k = speed
        elif v_k < -speed and v_k < 0:
            v_k = -speed
        if v_z_k > speed and v_z_k > 0:
            v_z_k = speed
        elif v_z_k < -speed and v_z_k < 0:
            v_z_k = -speed
        w_k = 1*(math.atan2(-y_k,-x_k) - theta_k + math.pi)


        new_x = self.position[0] + v_k*math.cos(theta_k + (w_k / 2))
        new_y = self.position[1] + v_k*math.sin(theta_k + (w_k / 2))
        new_z = self.position[2] + v_z_k
        new_theta = self.theta_k + w_k
        self.position = (new_x, new_y, new_z)
        self.theta_k = new_theta
        logging.debug("Moved to position %s %s %s" %(self.position[0], self.position[1], self.position[2]))