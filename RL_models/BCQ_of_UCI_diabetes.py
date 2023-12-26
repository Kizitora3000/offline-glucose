import os
import json
import numpy as np 
import copy, random, torch, gym, pickle
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from utils import test_algorithm, create_graph, unpackage_replay
from utils.evaluation import create_only_BG_graph

class FC_Q(nn.Module):
	def __init__(self, state_dim, num_actions):
		super(FC_Q, self).__init__()
		self.q1 = nn.Linear(state_dim, 256)
		self.q2 = nn.Linear(256, 256)
		self.q3 = nn.Linear(256, num_actions)

		self.i1 = nn.Linear(state_dim, 256)
		self.i2 = nn.Linear(256, 256)
		self.i3 = nn.Linear(256, num_actions)		


	def forward(self, state):
		q = F.relu(self.q1(state))
		q = F.relu(self.q2(q))

		i = F.relu(self.i1(state))
		i = F.relu(self.i2(i))
		i = self.i3(i)
		return self.q3(q), F.log_softmax(i, dim=1), i

class BCQ_of_UCI_diabetes:
    def __init__(self, init_seed, patient_params, params, qtable_path, model_path):
        # ENVIRONMENT
        self.params = params
        self.env_name = patient_params["env_name"]       
        self.folder_name = patient_params["folder_name"]   
        self.replay_name = patient_params["replay_name"]   
        self.bas = patient_params["u2ss"] * (patient_params["BW"] / 6000) * 3
        self.env = gym.make(self.env_name)
        self.action_size, self.state_size = 1, 11
        self.params["state_size"] = self.state_size
        self.sequence_length = 80   
        self.data_processing = "condensed"   
        self.device = params["device"]         
        
        # HYPERPARAMETERS
        self.batch_size = 256
        self.policy_arch = '256-256'
        self.qf_arch = '256-256'
        self.policy_log_std_multiplier = 1.0
        self.policy_log_std_offset = -1.0
        self.discount = 0.99
        self.alpha_multiplier = 1.0
        self.target_entropy = 0.0
        self.policy_lr = 3e-4
        self.qf_lr = 3e-4
        self.soft_target_update_rate = 5e-3
        self.target_update_period = 1
        self.cql_n_actions = 10
        self.cql_temp = 1.0
        self.cql_min_q_weight = 5.0
        self.cql_clip_diff_min = -np.inf
        self.cql_clip_diff_max = np.inf     

        # DISPLAY
        self.training_timesteps = params["training_timesteps"]
        self.training_progress_freq = int(self.training_timesteps // 10)
        
        # SEEDING
        self.train_seed = init_seed # use seeds 1, 2, 3        
        self.env.seed(self.train_seed) 
        np.random.seed(self.train_seed)
        torch.manual_seed(self.train_seed)  
        random.seed(self.train_seed)    
        
        # MEMORY
        self.memory_size = self.training_timesteps 
        self.memory = deque(maxlen=self.memory_size)

        with open(qtable_path, 'r') as f:
            self.Qtable = json.load(f)

        state_dim = 1
        num_actions=63
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = FC_Q(state_dim, num_actions).to(self.device)
        self.model.load_state_dict(torch.load(model_path))

    def select_action(self, state):
        """
        state = int(state)
        # 最大値を取得
        max_val = max(self.Qtable[state])

        # 最大値が-1e10であればランダムなインデックスを返す
        if max_val == -1e10:
            return np.random.choice(len(self.Qtable[state]))

        # 最大値のインデックスを取得
        max_indices = [i for i, val in enumerate(self.Qtable[state]) if val == max_val]
        
        # 最大値が複数存在する場合、ランダムに1つのインデックスを選ぶ
        # return np.random.choice(max_indices)
        """

        state_int = int(state)
        state = torch.tensor([[state]], dtype=torch.float32)
        _, imt, i = self.model(state)
        q = torch.tensor(self.Qtable[state_int], dtype=torch.float32)
        imt = imt.exp()
        threshold=0.3
        imt = (imt/imt.max(1, keepdim=True)[0] > threshold).float()
        # Use large negative number to mask actions from argmax
        action = (imt * q + (1. - imt) * -1e8).argmax(1)

        return action.item()


    def test_model(self, input_seed=0, input_max_timesteps=4800):
        # TESTING -------------------------------------------------------------------------------------------- 
        
        # initialise the environment
        env = gym.make(self.env_name)  

        # load the replay buffer
        with open("./Replays/" + self.replay_name + ".txt", "rb") as file:
            trajectories = pickle.load(file)  
        
        # Process the replay --------------------------------------------------

        # unpackage the replay
        self.memory, self.state_mean, self.state_std, self.action_mean, self.action_std, _, _ = unpackage_replay(
            trajectories=trajectories, empty_replay=self.memory, data_processing=self.data_processing, sequence_length=self.sequence_length
        )

        # update the parameters
        self.action_std = 1.75 * self.bas * 0.25 / (self.action_std / self.bas)
        self.params["state_mean"], self.params["state_std"]  = self.state_mean, self.state_std
        self.params["action_mean"], self.params["action_std"] = self.action_mean, self.action_std
        
        test_seed, max_timesteps = input_seed, input_max_timesteps

        # test the algorithm's performance vs pid algorithm
        rl_reward, rl_bg, rl_action, rl_insulin, rl_meals, pid_reward, pid_bg, pid_action = test_algorithm(
            env=env, agent_action=self.select_action, seed=test_seed, max_timesteps=max_timesteps,
            sequence_length=self.sequence_length, data_processing=self.data_processing, 
            pid_run=False, params=self.params, qleaning=True
        )
         
        # display the results
        # create_graph(
        create_only_BG_graph(
            rl_reward=rl_reward, rl_blood_glucose=rl_bg, rl_action=rl_action, rl_insulin=rl_insulin,
            rl_meals=rl_meals, pid_reward=pid_reward, pid_blood_glucose=pid_bg, 
            pid_action=pid_action, params=self.params
        )

    def calc_rl_blood_glucose(self, input_seed=0, input_max_timesteps=4800):
        # TESTING -------------------------------------------------------------------------------------------- 
        
        # initialise the environment
        env = gym.make(self.env_name)  

        # load the replay buffer
        with open("./Replays/" + self.replay_name + ".txt", "rb") as file:
            trajectories = pickle.load(file)  
        
        # Process the replay --------------------------------------------------

        # unpackage the replay
        self.memory, self.state_mean, self.state_std, self.action_mean, self.action_std, _, _ = unpackage_replay(
            trajectories=trajectories, empty_replay=self.memory, data_processing=self.data_processing, sequence_length=self.sequence_length
        )

        # update the parameters
        self.action_std = 1.75 * self.bas * 0.25 / (self.action_std / self.bas)
        self.params["state_mean"], self.params["state_std"]  = self.state_mean, self.state_std
        self.params["action_mean"], self.params["action_std"] = self.action_mean, self.action_std
        
        test_seed, max_timesteps = input_seed, input_max_timesteps

        # test the algorithm's performance vs pid algorithm
        rl_reward, rl_bg, rl_action, rl_insulin, rl_meals, pid_reward, pid_bg, pid_action = test_algorithm(
            env=env, agent_action=self.select_action, seed=test_seed, max_timesteps=max_timesteps,
            sequence_length=self.sequence_length, data_processing=self.data_processing, 
            pid_run=False, params=self.params, qleaning=True
        )

        return rl_bg