import os

directory_path1 = './Replays'
if not os.path.exists(directory_path1):
    os.makedirs(directory_path1)

directory_path2 = './Models'
if not os.path.exists(directory_path2):
    os.makedirs(directory_path2)

# INITIALISE THE ENVIRONMENT ----------------------

import numpy as np

from utils import create_env

# Set the parameters for the meal scenario
prob = [0.95, 0.1, 0.95, 0.1, 0.95, 0.1]
time_lb = np.array([5, 9, 10, 14, 16, 20])
time_ub = np.array([9, 10, 14, 16, 20, 23])
time_mu = np.array([7, 9.5, 12, 15, 18, 21.5])
time_sigma = np.array([30, 15, 30, 15, 30, 15])
amount_mu = [50, 15, 70, 15, 90, 30]
amount_sigma = [10, 5, 10, 5, 10, 5]   
schedule=[prob, time_lb, time_ub, time_mu, time_sigma, amount_mu, amount_sigma]

# Incorporate the schedule into the environment
create_env(schedule=schedule)
# SPECIFY THE PARAMETERS -----------------------

from utils import get_params

# Get the parameters for a specified patient
patient_params = get_params()["adult#1"]
bas = patient_params["u2ss"] * (patient_params["BW"] / 6000) * 3

# Set the parameters
params = {
    
    # Environmental
    "state_size": 3,
    "basal_default": bas, 
    "target_blood_glucose": 144.0 ,
    "days": 10,    
    
    # PID and Bolus
    "carbohydrate_ratio": patient_params["carbohydrate_ratio"],
    "correction_factor":  patient_params["correction_factor"],
    "kp": patient_params["kp"],
    "ki": patient_params["ki"],
    "kd": patient_params["kd"],
    
    # RL 
    "training_timesteps": int(1e5),
    "device": "cpu",
    "rnn": None
}

# COLLECT THE DATA --------------------------------

from utils import fill_replay_split

import gym

# initialise the environment
env = gym.make(patient_params["env_name"])

# Fill the replay
full_replay = fill_replay_split(
    env=env, 
    replay_name=patient_params["replay_name"],
    data_split=0.0,
    noise=True,
    bolus_noise=0.1,
    seed=0,
    params=params
)
