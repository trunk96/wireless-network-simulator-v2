import random
import math
import logging
import numpy.random


logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)



class MultiPathUserEquipment:
    MAX_QUEUE = 1.0 #Mbit
    def __init__(self, env, ue_id, initial_data_rate, starting_position, speed = 0, direction = 0, random = False, _lambda_c = None, _lambda_d = None):
        self.ue_id = ue_id
        self.queue = 0
        self.queue_out = False
        self.input_data_rate = initial_data_rate
        self.current_position = starting_position
        self.env = env
        self.speed = speed * self.env.get_sampling_time()
        self.direction = direction
        self._lambda_c = _lambda_c
        if self._lambda_c != None:
            self.time_to_wait = numpy.random.poisson(self._lambda_c)
        else:
            self.time_to_wait = None
        self.data_generation_status = False # if generating data -> True, if not generating data -> False
        self._lambda_d = _lambda_d
        self.last_time = 0
        self.random = random
        self.output_data_rate = {}

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
            return list(self.bs_data_rate_allocation.keys())

    def advertise_connection(self):
        if self.queue > 0 or self.data_generation_status:
            # we have something to send, so please send it
            self.env.advertise_connection(self.ue_id)
        return

    def update_queue(self):
        total_output_data_rate = 0
        for elem in self.output_data_rate:
            total_output_data_rate += min(self.output_data_rate[elem], self.bs_data_rate_allocation[elem])  # do not transmit more than what is allocated           
        if self.data_generation_status == True:
            self.queue += self.sampling_time * (self.input_data_rate - total_output_data_rate)
        else:
            self.queue += self.sampling_time * total_output_data_rate
        self.queue_out = False
        if self.queue > MultiPathUserEquipment.MAX_QUEUE+0.1:
            self.queue = MultiPathUserEquipment.MAX_QUEUE
            self.queue_out = True
    
    def generate_input_data_rate(self):
        if self.time_to_wait == None:
            return
        if self.last_time >= self.time_to_wait:
            self.last_time = 0
            if self.data_generation_status == False:
                # it is time to start generating data
                self.data_generation_status = True
                if self._lambda_d != None:
                    self.time_to_wait = numpy.random.poisson(self._lambda_d)
                else:
                    self.time_to_wait = None
            else:
                self.data_generation_status = False
                if self._lambda_c != None:
                    self.time_to_wait = numpy.random.poisson(self._lambda_c)
                else:
                    self.time_to_wait = None
        else:
            self.last_time += 1
    
    def get_current_input_data_rate(self):
        if self.data_generation_status:
            return self.input_data_rate
        return 0
    
    def get_queue(self):
        return self.queue

    def step(self):
        self.update_queue()
        self.move()
        self.advertise_connection()
        self.generate_input_data_rate()
        self.output_data_rate.clear()
        return
    
    def connect_bs(self, bs_list):
        rsrp = self.measure_rsrp()
        if len(rsrp) == 0:
            return
        for bs in bs_list:
            self.connect_(bs, rsrp)
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
        return self.connect_(best_bs, rsrp)

    def connect_(self, bs, rsrp):
        if len(self.bs_data_rate_allocation) == 0:
            # no BS connected
            bs = self.env.bs_by_id(bs)
            actual_data_rate = bs.connect(self.ue_id, self.output_data_rate[bs.get_id()], rsrp)
            self.bs_data_rate_allocation[bs.get_id()] = actual_data_rate
            logging.info("UE %s not connected is now connected to BS %s with data rate %s", self.ue_id, bs.get_id(), actual_data_rate)
        else:
            current_bs = self.get_current_bs()
            if bs not in current_bs:
                #self.disconnect()
                bs = self.env.bs_by_id(bs)
                actual_data_rate = bs.connect(self.ue_id, self.output_data_rate[bs.get_id()], rsrp)
                self.bs_data_rate_allocation[bs.get_id()] = actual_data_rate
                logging.info("UE %s connected to BS %s with data rate %s", self.ue_id, bs.get_id(), actual_data_rate) 
            else:
                current_bs = self.env.bs_by_id(bs)
                actual_data_rate = current_bs.update_connection(self.ue_id, self.output_data_rate[current_bs.get_id()], rsrp)
                logging.info("UE %s updated to BS %s with data rate %s --> %s", self.ue_id, current_bs.get_id(), self.bs_data_rate_allocation[current_bs.get_id()], actual_data_rate)
                self.bs_data_rate_allocation[current_bs.get_id()] = actual_data_rate
        return

    def disconnect(self, bs):
        if bs in self.get_current_bs():
            self.env.bs_by_id(bs).disconnect(self.ue_id)
            del self.bs_data_rate_allocation[bs]

    def disconnect_all(self):
        bs_list = self.get_current_bs()
        if bs_list != None:
            for bs in bs_list:
                self.disconnect(bs)
    
    def requested_disconnect(self):
        # this is called if the env or the BS requested a disconnection
        current_bs = self.get_current_bs()
        del self.bs_data_rate_allocation[current_bs]


        

    