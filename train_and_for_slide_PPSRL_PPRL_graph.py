# INITIALISE THE ENVIRONMENT ----------------------

import numpy as np
from RL_models.Qlearning import qlearning
from utils import create_env
import matplotlib.pyplot as plt
from utils.general import is_in_range

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

def create_only_BG_graph(rl_blood_glucose1, rl_blood_glucose2, rl_blood_glucose3, params):
    
    # Unpack the params

    # Diabetes
    basal_default = params.get("basal_default")    
    hyper_threshold = params.get("hyper_threshold", 180) 
    sig_hyper_threshold = params.get("sig_hyper_threshold", 250)
    hypo_threshold = params.get("hypo_threshold ", 70)
    sig_hypo_threshold = params.get("sig_hypo_threshold ", 54)

    
    x = list(range(len(rl_blood_glucose1)))
    
    plt.rcParams['font.family'] = 'MS Gothic'
    plt.rcParams['font.size'] = 28
    
    # define the hypo, eu and hyper regions
    # plt.axhspan(180, 300, color='lightcoral', alpha=0.6, lw=0)
    # plt.axhspan(70, 180, color='#c1efc1', alpha=1.0, lw=0)
    plt.axhspan(0, 70, color='skyblue', alpha=0.6, lw=0)
    
    # plot the blood glucose values
    # plt.plot(x, rl_blood_glucose1, label='SRL', color='darkorange', alpha=1, linestyle = "dashed")
    plt.plot(x, rl_blood_glucose2, label='リスク関数を用いたプライバシ保護安全強化学習', color='darkblue', alpha=1, linestyle = "solid", linewidth=3.0)
    plt.plot(x, rl_blood_glucose3, label='プライバシ保護強化学習', color='#fa5502', alpha=0.7, linestyle = "dashed", linewidth=3.0)
    # #fa5502 #de4c02
    plt.legend()
    
    # specify the limits and the axis labels
    plt.axis(ymin=50, ymax=280)
    plt.axis(xmin=0.0, xmax=len(rl_blood_glucose1))
    plt.ylabel("血糖値 [mg/dL]")
    plt.xlabel("時間 [日]")
    # plt.title('各方策における血糖値制御結果')

    positions = range(0, len(rl_blood_glucose1) + 1, 480)
    labels = [str(int(pos/480)) for pos in positions] # 実際の位置を20で割ってラベルを作成
    # 横軸の目盛りをカスタマイズ
    plt.xticks(positions, labels)

    # plt.subplots_adjust(left=0.1, right=0.98, top=0.97, bottom=0.15)

    plt.show()

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

import torch
if torch.cuda.is_available():
    params["device"] = "cuda"

agent1 = qlearning(
    init_seed=0,
    patient_params=patient_params,
    params=params,
    data_file_path="srl_data.json"
)

agent2 = qlearning(
    init_seed=0,
    patient_params=patient_params,
    params=params,
    data_file_path="pprl_srl_data.json"
)

agent3 = qlearning(
    init_seed=0,
    patient_params=patient_params,
    params=params,
    data_file_path="pprl_rl_data.json"
)

# Train the agent
ag1_rl_bg = agent1.calc_rl_blood_glucose()
ag2_rl_bg = agent2.calc_rl_blood_glucose()
ag3_rl_bg = agent3.calc_rl_blood_glucose()

create_only_BG_graph(ag1_rl_bg, ag2_rl_bg, ag3_rl_bg, params)