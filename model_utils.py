import gymnasium as gym
import numpy as np

from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

def train_model(time_steps,init_exploration=1,fin_exploration=0.1,learning_rate=0.0001):
	model = DQN(
    "MlpPolicy",
    "LunarLander-v2",
    exploration_initial_eps=init_exploration,
    learning_rate=learning_rate,
    exploration_final_eps=fin_exploration,
    target_update_interval=250,
	)
	model.learn(total_timesteps=time_steps)
	return model