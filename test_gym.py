from json import load

from pkg_resources import load_entry_point
from wns2.basestation.satellitebasestation import SatelliteBaseStation
from wns2.gym.cac_env import CACGymEnv
import numpy.random as random
import logging
import lexicographicqlearning
import signal
import numpy as np

logger = logging.getLogger()
logger.setLevel(level=logging.WARNING)


x_lim = 1000
y_lim = 600
n_ue = 100 
class_list = []
for i in range(n_ue):
    class_list.append(i % 3)

quantization = 6

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
    "max_bitrate": 1000}
] 
'''    
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
    "max_bitrate": 1000}]'''

sat_parm = [{"pos": (250, 500, 35786000)}]
env = CACGymEnv(x_lim, y_lim, class_list, terr_parm, sat_parm, datarate = 50, quantization=quantization)
learner = lexicographicqlearning.LexicographicQTableLearner(env, "CAC_Env", [0.075, 0.10, 0.15])

def exit_handler(signum, frame):
    res = input("Ctrl-c was pressed, do you want to save your current model? y/n ")
    if res == "y":
        global learner
        learner.save_model()
        exit(1)
    else: 
        exit(1)

signal.signal(signal.SIGINT, exit_handler)

#learner.train(train_episodes=100000)
#learner.save_model()
learner.load_model("CAC_Env", path="saved_models/100UE_50mbps_5BS_100000_1000/")
print("Model loaded")
LQL_rewards = learner.test(test_episodes=1000)
print("Model tested")

LL_rewards = ([], [])
for i in range(1000):
    curr_state = env.reset()
    total_reward = 0
    total_constraint_reward = np.zeros(3)
    for j in range(1000):
        load_levels = np.zeros(len(terr_parm)+len(sat_parm))
        reminder = curr_state
        print(curr_state)
        for k in range(len(load_levels)):
            load_levels[k] = reminder % quantization
            reminder = reminder // quantization
        print(f"Load Level: {load_levels}")
        action = np.argmin(load_levels)
        print(f"Action chosen: {action}")

        new_state, reward, done, info = env.step(action+1)
        curr_state = new_state
        for _ in range(len(info)):
            reward_constr = info[_]
            if reward_constr == -1:
                reward_constr = 0
            total_constraint_reward[_] += reward_constr
        total_reward += reward
    LL_rewards[0].append(total_reward)
    LL_rewards[1].append(total_constraint_reward)
np.save("LQL_rewards", LQL_rewards)
np.save("LL_rewards", LL_rewards)

print(np.mean(LQL_rewards[0]))
print(np.mean(LL_rewards[0]))





'''counter = 10
while counter != 0:
    #action = random.randint(0, len(terr_parm)+len(sat_parm))
    action = [1, 3]
    print(env.step(action))
    counter -= 1'''

