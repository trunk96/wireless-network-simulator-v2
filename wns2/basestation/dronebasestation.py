from wns2.basestation.nrbasestation import NRBaseStation
import math
from scipy import constants
from wns2.environment import environment
from wns2.pathloss import costhata
import logging

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)


class DroneBaseStation(NRBaseStation):   
    def __init__(self, env, bs_id, position, carrier_frequency, total_bandwidth, numerology, max_data_rate = None, antenna_power = 20, antenna_gain = 16, feeder_loss = 3, pathloss = None):
        super().__init__(env, bs_id, position, carrier_frequency, total_bandwidth, numerology, max_data_rate = None, antenna_power = 20, antenna_gain = 16, feeder_loss = 3, pathloss = None)
        self.theta_k = 0
        self.bs_type = "drone"
        self.integral_error = [0,0,0]
        self.last_error = [0,0,0]
        return

    def move_PID(self, destination, speed, k_p = 1, k_i = 0, k_d = 0):
        e_x = destination[0] - self.position[0]
        e_y = destination[1] - self.position[1]
        e_z = destination[2] - self.position[2]
        max_speed = speed * self.env.get_sampling_time()
        self.integral_error[0] += e_x
        self.integral_error[1] += e_y
        self.integral_error[2] += e_z
        v_ref_x = k_p * e_x + k_i * self.integral_error[0] + k_d * (e_x - self.last_error[0])
        v_ref_y = k_p * e_y + k_i * self.integral_error[1] + k_d * (e_y - self.last_error[1])
        v_ref_z = k_p * e_z + k_i * self.integral_error[2] + k_d * (e_z - self.last_error[2])
        self.last_error = [e_x, e_y, e_z]
        mod_v = v_ref_x**2 +v_ref_y**2 + v_ref_z**2
        if  mod_v > speed **2:
            alpha = speed / math.sqrt(mod_v)
            v_ref_x *= alpha
            v_ref_y *= alpha
            v_ref_z *= alpha
        new_x = self.position[0] + v_ref_x
        new_y = self.position[1] + v_ref_y
        new_z = self.position[2] + v_ref_z
        self.position = (new_x, new_y, new_z)
        return


    def move_unicycle(self, destination, speed):
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

    def move(self, destination, speed):
        self.move_PID(destination, speed)