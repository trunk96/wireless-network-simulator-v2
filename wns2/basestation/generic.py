# skeleton of Base Station classes
class BaseStation:
    def __init__(self):
        return
    
    def get_position(self):
        return NotImplementedError
    
    def get_carrier_frequency(self):
        return NotImplementedError
    
    def get_bs_type(self):
        return NotImplementedError
    
    def get_id(self):
        return NotImplementedError
    
    def compute_rsrp(self, ue):
        return NotImplementedError

    def get_rbur(self):
        return NotImplementedError
    
    def connect(self, ue_id, desired_data_rate, rsrp):
        return NotImplementedError
    
    def disconnect(self, ue_id):
        return NotImplementedError

    def update_connection(self, ue_id, desired_data_rate, rsrp):
        return NotImplementedError
    
    def step(self):
        return NotImplementedError