import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
import streamlit as st
import time


env = gym.make("LunarLander-v2",render_mode='rgb_array')

st.title('Lunar Lander')

st.write('#### Train a model to play the lunar lander')

agents = []
n = len(agents) if len(agents) else 2
cols = st.columns(n)
containers = [st.empty() for i in range(n)]


vec_env = [None for i in range(n)]
obs = [None for i in range(n)]
rewards = [None for i in range(n)]
dones = [None for i in range(n)]
info = [None for i in range(n)]


for i,col in enumerate(cols):
	with col:
		containers[i] = st.empty()



if st.checkbox('Add Random Agent'):
	agents.append(DQN(
			    "MlpPolicy",
			    "LunarLander-v2",
			    verbose=1,
			    exploration_final_eps=0.1,
			    target_update_interval=250))

if st.checkbox('Add Trained Agent'):
	agents.append(DQN.load("dqn_lunar",env=env))

with st.form(key='main_form'):
        col1, col2, col3 = st.columns(3)
        with col1:
            start_button = st.form_submit_button(label='Start')
        with col2:    
            stop_button = st.form_submit_button(label='Stop')
        with col3:
        	step_button = st.form_submit_button(label='Step') 

fps = st.number_input('FPS',min_value=10,max_value=240,step=5)
flag = False

if start_button:
	for i,model in enumerate(agents):
		vec_env[i] = model.get_env()
		obs[i] = vec_env[i].reset()
	for i in range(fps*5):
		for j,model in enumerate(agents):
			vec_env[j] = model.get_env()
			action, _states = model.predict(obs[j], deterministic=True)
			obs[j], rewards[j], dones[j], info[j] = vec_env[j].step(action)
			img = vec_env[j].render("rgb_array")
			with cols[j]:
				containers[j].image(img, use_column_width=True)
			time.sleep(1/fps)
		if stop_button:
			flag = False
		    break 	


if step_button:
	if flag = False:
		flag = True
		for i,model in enumerate(agents):
		vec_env[i] = model.get_env()
		obs[i] = vec_env[i].reset()
	else:
		for j,model in enumerate(agents):
			vec_env[j] = model.get_env()
			action, _states = model.predict(obs[j], deterministic=True)
			obs[j], rewards[j], dones[j], info[j] = vec_env[j].step(action)
			img = vec_env[j].render("rgb_array")
			with cols[j]:
				containers[j].image(img, use_column_width=True)


				
