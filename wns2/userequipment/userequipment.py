import random
import math
import logging
import numpy.random


logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)


class UserEquipment:
   
    def __init__(self, env, ue_id, initial_data_rate, starting_position, speed = 0, direction = 0, random = False, _lambda_c = None, _lambda_d = None):
        self.ue_id = ue_id
        self.data_rate = initial_data_rate
        self.current_position = starting_position
        self.env = env
        self.speed = speed
        self.direction = direction
        self._lambda_c = _lambda_c
        if self._lambda_c != None:
            self.connection_time_to_wait = numpy.random.poisson(self._lambda_c)
        self._lambda_d = _lambda_d
        self.last_time = 0
        self.random = random

        self.sampling_time = self.env.get_sampling_time()

        self.bs_data_rate_allocation = {}

    def get_position(self):
        return self.current_position
    
    def get_id(self):
        return self.ue_id

    def move(self):
        if self.speed == 0:
            return
        if self.random == True:
            return self.random_move()
        else:
            return self.line_move()
    
    def random_move(self):
        val = random.randint(1, 4)
        size = random.randint(0, math.floor(self.speed*self.sampling_time))
        x_lim = self.env.get_x_limit()
        y_lim = self.env.get_y_limit()
        if val == 1: 
            if (self.current_position[0] + size) > 0 and (self.current_position[0] + size) < x_lim:
                self.current_position = (self.current_position[0] + size, self.current_position[1], self.current_position[2])
        elif val == 2: 
            if (self.current_position[0] - size) > 0 and (self.current_position[0] - size) < x_lim:
                self.current_position = (self.current_position[0] - size, self.current_position[1], self.current_position[2])
        elif val == 3: 
            if (self.current_position[1] + size) > 0 and (self.current_position[1] + size) < y_lim:
                self.current_position = (self.current_position[0], self.current_position[1] + size, self.current_position[2])
        else: 
            if (self.current_position[1] - size) > 0 and (self.current_position[1] - size) < y_lim:
                self.current_position = (self.current_position[0], self.current_position[1] - size, self.current_position[2])
        return self.current_position

    def line_move(self):
        new_x = self.current_position[0]+self.speed*self.sampling_time*math.cos(math.radians(self.direction))
        new_y = self.current_position[1]+self.speed*self.sampling_time*math.sin(math.radians(self.direction))
        x_lim = self.env.get_x_limit()
        y_lim = self.env.get_y_limit()
        # bounce with the same incident angle if a sideo or a corner is reached
        if ((self.current_position[0] != 0 and math.tan(math.radians(self.direction)) == (self.current_position[1])/(self.current_position[0])) or (self.direction == 270)) :
            if new_x <= 0 and new_y <= 0:
                # bottom left corner bouncing
                self.direction = 270 - self.direction
                dist = math.sqrt((new_x)**2 + (new_y)**2)
                new_x = dist*math.cos(math.radians(self.direction))
                new_y = dist*math.sin(math.radians(self.direction))
        elif ((x_lim - self.current_position[0] != 0 and math.tan(math.radians(self.direction)) == (y_lim - self.current_position[1])/(x_lim - self.current_position[0])) or (self.direction == 90)) :
            if new_x >= x_lim and new_y >= y_lim :
                # top right corner bouncing
                self.direction = 270 - self.direction
                dist = math.sqrt((x_lim-new_x)**2 + (y_lim-new_y)**2)
                new_x = x_lim + dist*math.cos(math.radians(self.direction))
                new_y = y_lim + dist*math.sin(math.radians(self.direction))
        elif ((self.current_position[0] != 0 and math.tan(math.radians(self.direction)) == (y_lim - self.current_position[1])/(self.current_position[0])) or (self.direction == 90)) :
            if new_x <= 0 and new_y >= y_lim:
                # top left corner bouncing
                self.direction = 450 - self.direction
                dist = math.sqrt((new_x)**2 + (y_lim-new_y)**2)
                new_x = dist*math.cos(math.radians(self.direction))
                new_y = y_lim + dist*math.sin(math.radians(self.direction))
        elif ((x_lim - self.current_position[0] != 0 and math.tan(math.radians(self.direction)) == (self.current_position[1])/(x_lim - self.current_position[0])) or (self.direction == 270)) :
            if new_x >= x_lim and new_y <= 0 :
                # bottom right corner bouncing
                self.direction = 450 - self.direction
                dist = math.sqrt((new_x)**2 + (y_lim-new_y)**2)
                new_x = dist*math.cos(math.radians(self.direction))
                new_y = y_lim + dist*math.sin(math.radians(self.direction))
        elif new_y <= 0:
            # bottom side bouncing
            new_y = 0 - new_y
            self.direction = 360 - self.direction
            if new_x <= 0:
                # there is another bouncing on the left side
                new_x = 0 - new_x
                self.direction = 180 - self.direction
            elif new_x >= x_lim:
                # there is another bouncing on the right side
                new_x = 2*x_lim - new_x
                self.direction = 180 - self.direction
        elif new_x <= 0 :
            # left side bouncing
            new_x = 0 - new_x
            self.direction = 180 - self.direction
            if new_y <= 0:
                # there is another bouncing on the bottom side
                new_y = 0 - new_y
                self.direction = - self.direction
            elif new_y >= y_lim:
                # there is another bouncing on the top side
                new_y = 2*y_lim - new_y
                self.direction = - self.direction
        elif new_y >= y_lim:
            # top side bouncing
            new_y = 2*y_lim - new_y
            self.direction = 360 - self.direction
            if new_x <= 0:
                # there is another bouncing on the left side
                new_x = 0 - new_x
                self.direction = 180 - self.direction
            elif new_x >= x_lim:
                # there is another bouncing on the left side
                new_x = 2*x_lim - new_x
                self.direction = 180 - self.direction
        elif new_x >= x_lim:
            # right side bouncing
            new_x = 2*x_lim - new_x
            self.direction =  180 - self.direction
            if new_y <= 0:
                # there is another bouncing on the bottom side
                new_y = 0 - new_y
                self.direction = -self.direction
            elif new_y >= y_lim:
                # there is another bouncing on the top side
                new_y = 2*y_lim - new_y
                self.direction = - self.direction

        self.current_position = (new_x, new_y, self.current_position[2])
        self.direction = self.direction % 360
        return self.current_position
    
    def measure_rsrp(self):
        # measure RSRP together with the BS
        # the result is in dB
        return self.env.compute_rsrp(self)
    
    def get_current_bs(self):
        if len(self.bs_data_rate_allocation) == 0:
            return None
        else:
            return list(self.bs_data_rate_allocation.keys())[0]
    
    def step(self):
        self.move()
        if len(self.bs_data_rate_allocation) == 0:
            # no BS connected, decide if it is time to connect
            if self._lambda_c == None:
                self.connect_max_rsrp()
            elif self.last_time >= self.connection_time_to_wait:
                self.last_time = 0
                self.connect_max_rsrp()
                if self._lambda_d != None:
                    self.disconnection_time_to_wait = numpy.random.poisson(self._lambda_d)
            else:
                self.last_time += 1
        else:
            if self._lambda_d == None:
                self.connect_max_rsrp()
            elif self.last_time >= self.disconnection_time_to_wait:
                self.last_time = 0
                self.disconnect()
                if self._lambda_c != None:
                    self.connection_time_to_wait = numpy.random.poisson(self._lambda_c)
            else:
                self.last_time += 1
                self.connect_max_rsrp()

        return

    def connect_max_rsrp(self):
        rsrp = self.measure_rsrp()
        if len(rsrp) == 0:
            return
        best_bs = None
        max_rsrp = -200
        for elem in rsrp:
            if rsrp[elem] > max_rsrp:
                best_bs = elem
                max_rsrp = rsrp[elem]
        if len(self.bs_data_rate_allocation) == 0:
            # no BS connected
            best_bs = self.env.bs_by_id(best_bs)
            actual_data_rate = best_bs.connect(self.ue_id, self.data_rate, rsrp)
            self.bs_data_rate_allocation[best_bs.get_id()] = actual_data_rate
            logging.info("UE %s connected to BS %s with data rate %s", self.ue_id, best_bs.get_id(), actual_data_rate)
        else:
            current_bs = self.get_current_bs()
            if current_bs != best_bs:
                self.disconnect()
                best_bs = self.env.bs_by_id(best_bs)
                actual_data_rate = best_bs.connect(self.ue_id, self.data_rate, rsrp)
                self.bs_data_rate_allocation[best_bs.get_id()] = actual_data_rate
                logging.info("UE %s switched to BS %s with data rate %s", self.ue_id, best_bs.get_id(), actual_data_rate)
            else:
                current_bs = self.env.bs_by_id(current_bs)
                actual_data_rate = current_bs.update_connection(self.ue_id, self.data_rate, rsrp)
                logging.info("UE %s updated to BS %s with data rate %s --> %s", self.ue_id, current_bs.get_id(), self.bs_data_rate_allocation[current_bs.get_id()], actual_data_rate)
                self.bs_data_rate_allocation[current_bs.get_id()] = actual_data_rate
        return

    def disconnect(self):
        current_bs = self.get_current_bs()
        self.env.bs_by_id(current_bs).disconnect(self.ue_id)
        del self.bs_data_rate_allocation[current_bs]


        

    