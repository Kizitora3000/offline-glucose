import numpy as np
from utils import create_env, get_params
from RL_models.Qlearning import qlearning
import matplotlib.pyplot as plt
import os

def create_only_BG_graph(rl_blood_glucose1, rl_blood_glucose2, params):
    
    # Unpack the params

    # Diabetes
    basal_default = params.get("basal_default")    
    hyper_threshold = params.get("hyper_threshold", 180) 
    sig_hyper_threshold = params.get("sig_hyper_threshold", 250)
    hypo_threshold = params.get("hypo_threshold ", 70)
    sig_hypo_threshold = params.get("sig_hypo_threshold ", 54)

    plt.figure(figsize=(20, 10))  
    
    x = list(range(len(rl_blood_glucose1)))
    
    plt.rcParams['font.family'] = 'MS Gothic'
    plt.rcParams['font.size'] = 28

    
    # define the hypo, eu and hyper regions
    plt.axhspan(0, 70, color='skyblue', alpha=0.6, lw=0)
    
    # plot the blood glucose values
    plt.plot(x, rl_blood_glucose1, label='MKPPSRL_RL', color='darkblue', alpha=1, linestyle = "solid", linewidth=3.0)
    plt.plot(x, rl_blood_glucose2, label='TOM_RL', color='#fa5502', alpha=0.7, linestyle = "dashed", linewidth=3.0)
    plt.legend(loc='upper right', ncol=2)

    # specify the limits and the axis labels
    plt.axis(ymin=50, ymax=280)
    plt.axis(xmin=0.0, xmax=len(rl_blood_glucose1))
    plt.ylabel("blood glucose level [mg/dL]")
    plt.xlabel("time [day]")

    positions = range(0, len(rl_blood_glucose1) + 1, 480)
    labels = [str(int(pos/480)) for pos in positions] # 実際の位置を20で割ってラベルを作成
    # 横軸の目盛りをカスタマイズ
    plt.xticks(positions, labels)

    # plt.subplots_adjust(left=0.1, right=0.98, top=0.97, bottom=0.15)

    # if save:
    #    plt.savefig(f"result/2024_12_28/AM/MKPPSRL_RL_and_TOM_RL_state_dim_1_timestep_{mstep}.png")

    plt.show()

# ---------- 環境を初期化  ----------
prob = [0.95, 0.1, 0.95, 0.1, 0.95, 0.1]
time_lb = np.array([5, 9, 10, 14, 16, 20])
time_ub = np.array([9, 10, 14, 16, 20, 23])
time_mu = np.array([7, 9.5, 12, 15, 18, 21.5])
time_sigma = np.array([30, 15, 30, 15, 30, 15])
amount_mu = [50, 15, 70, 15, 90, 30]
amount_sigma = [10, 5, 10, 5, 10, 5]   
schedule=[prob, time_lb, time_ub, time_mu, time_sigma, amount_mu, amount_sigma]

create_env(schedule=schedule)

patient_params = get_params()["adult#1"]
bas = patient_params["u2ss"] * (patient_params["BW"] / 6000) * 3

# Set the parameters
params = {
    
    # Environmental
    "state_size": 11,
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

# Initialise the agent

parameters = {
    # Exploration
    "start_timesteps": 1e3,
    "initial_eps": 0.1,
    "end_eps": 0.1,
    "eps_decay_period": 1,
    # Evaluation
    "eval_freq": 1e2,
    "eval_eps": 0,
    # Learning
    "discount": 0.99,
    "buffer_size": 1e6,
    "batch_size": 100,
    "optimizer": "Adam",
    "optimizer_parameters": {
        "lr": 1e-3
    },
    "train_freq": 1,
    "polyak_target_update": True,
    "target_update_freq": 1,
    "tau": 0.005
}

state_dim = 1
num_actions = 63
device = "cpu"
BCQ_threshold=0.3

# デバック時に使用するtxtははじめに初期化
txt_name = "temp.txt"

if os.path.exists(txt_name):
    os.remove(txt_name)
"""
import sys
args = sys.argv

if len(args) < 2:
    print("Please select a timestep")
    exit()

timestep = int(args[1])

json_path1 = f"./dataset/2024_12_28/AM/SRL_state_dim_1_timestep_{timestep}.json"
json_path2 = f"./dataset/others/rl_data.json"
"""

agent1 = qlearning(
    init_seed=0,
    patient_params=patient_params,
    params=params,
    data_file_path="pprl_srl_data.json"
)

agent2 = qlearning(
    init_seed=0,
    patient_params=patient_params,
    params=params,
    data_file_path="pprl_rl_data.json"
)

rl_blood_glucose1 = agent1.calc_rl_blood_glucose()
rl_blood_glucose2 = agent2.calc_rl_blood_glucose()

save = False

create_only_BG_graph(rl_blood_glucose1, rl_blood_glucose2, params)