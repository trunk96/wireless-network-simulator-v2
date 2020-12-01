import random
import math

MAX_STEP = 2000

class UserEquipment:
   
    def __init__(self, env, ue_id, initial_data_rate, starting_position, speed = 0, direction = 0, random = False, _lambda = 1/15):
        self.ue_id = ue_id
        self.data_rate = initial_data_rate
        self.current_position = starting_position
        self.env = env
        self.speed = self.speed
        self.direction = self.direction
        self._lambda = _lambda
        self.random = random

        self.bs_data_rate_allocation = {}

    def get_position(self):
        return self.current_position

    def move(self):
        if self.speed == 0:
            return
        if self.random == False:
            return self.random_move()
        else:
            return self.line_move()
    
    def random_move(self):
        val = random.randint(1, 4)
        size = random.randint(0, MAX_STEP) 
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
        new_x = self.current_position[0]+self.speed*math.cos(math.radians(self.direction))
        new_y = self.current_position[1]+self.speed*math.sin(math.radians(self.direction))
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
    
    def measure_rsrp(self, bs):
        # measure RSRP together with the BS
        # the result is in dB
        return bs.compute_rsrp(self)

    