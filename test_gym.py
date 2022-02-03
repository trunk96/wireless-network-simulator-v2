from wns2.basestation.satellitebasestation import SatelliteBaseStation
from wns2.gym.cac_env import CACGymEnv
import numpy.random as random
import logging
import lexicographicqlearning
import signal

logger = logging.getLogger()
logger.setLevel(level=logging.INFO)


x_lim = 1000
y_lim = 1000
n_ue = 50
class_list = []
for i in range(n_ue):
    class_list.append(i % 3)

terr_parm =[{"pos": (500, 500, 30),
    "freq": 800,
    "numerology": 1, 
    "power": 20,
    "gain": 16,
    "loss": 3,
    "bandwidth": 20,
    "max_bitrate": 1000},
    

    #BS2
    {"pos": (250, 300, 30),
    "freq": 1700,
    "numerology": 1, 
    "power": 20,
    "gain": 16,
    "loss": 3,
    "bandwidth": 40,
    "max_bitrate": 1000},

    #BS3
    {"pos": (500, 125, 30),
    "freq": 1900,
    "numerology": 1, 
    "power": 20,
    "gain": 16,
    "loss": 3,
    "bandwidth": 40,
    #15
    "max_bitrate": 1000},

    #BS4
    {"pos": (750, 300, 30),
    "freq": 2000,
    "numerology": 1, 
    "power": 20,
    "gain": 16,
    "loss": 3,
    "bandwidth": 25,
    "max_bitrate": 1000},
    
    #BS5
    {"pos": (750, 700, 30),
    "freq": 1700,
    "numerology": 1, 
    "power": 20,
    "gain": 16,
    "loss": 3,
    "bandwidth": 40,
    "max_bitrate": 1000},

    #BS6
    {"pos": (500, 875, 30),
    "freq": 1900,
    "numerology": 1, 
    "power": 20,
    "gain": 16,
    "loss": 3,
    "bandwidth": 40,
    "max_bitrate": 1000},

    #BS7
    {"pos": (250, 700, 30),
    "freq": 2000,
    "numerology": 1, 
    "power": 20,
    "gain": 16,
    "loss": 3,
    "bandwidth": 25,
    "max_bitrate": 1000}]

sat_parm = [{"pos": (250, 500, 35786000)}]
env = CACGymEnv(x_lim, y_lim, class_list, terr_parm, sat_parm)
learner = lexicographicqlearning.LexicographicQTableLearner(env, "CAC_Env", [0.075, 0.075, 0.075])

def exit_handler(signum, frame):
    res = input("Ctrl-c was pressed, do you want to save your current model? Y/n ")
    if res == "y":
        global learner
        learner.save_model()
        exit(1)

signal.signal(signal.SIGINT, exit_handler)

learner.train(train_episodes=500)
learner.test()
learner.save_model()



'''counter = 10
while counter != 0:
    #action = random.randint(0, len(terr_parm)+len(sat_parm))
    action = [1, 3]
    print(env.step(action))
    counter -= 1'''

