import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
import streamlit as st
import time

st.set_page_config(layout='wide')

agents = []
n = len(agents) if len(agents) else 2
env = gym.make("LunarLander-v2", render_mode='rgb_array')

if 'vec_env' not in st.session_state:
    st.session_state.vec_env = [None for i in range(n)]
if 'obs' not in st.session_state:
    st.session_state.obs = [None for i in range(n)]
if 'rewards' not in st.session_state:
    st.session_state.rewards = [None for i in range(n)]
if 'dones' not in st.session_state:
    st.session_state.dones = [None for i in range(n)]
if 'info' not in st.session_state:
    st.session_state.info = [None for i in range(n)]
if 'frames' not in st.session_state:
    st.session_state.frames = [[] for i in range(n)]

st.title('Lunar Lander')

st.write('#### Train a model to play the lunar lander')

cols = st.columns(n)
containers = [st.empty() for i in range(n)]

for i, col in enumerate(cols):
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
    agents.append(DQN.load("dqn_lunar", env=env))

with st.form(key='main_form'):
    col1, col2, col3 = st.columns(3)
    with col1:
        start_button = st.form_submit_button(label='Start')
    with col2:
        stop_button = st.form_submit_button(label='Stop')
    with col3:
        display_button = st.form_submit_button(label='Display')

fps = st.number_input('FPS', min_value=1, max_value=240, value=60, step=5)
time_steps = st.number_input('How long should agent play?', min_value = 10, max_value = 60, value = 10, step=5)

if start_button:
	with st.spinner("Agents are playing..."):
	    for i, model in enumerate(agents):
	        st.session_state.vec_env[i] = model.get_env()
	        st.session_state.obs[i] = st.session_state.vec_env[i].reset()
	        st.session_state.frames = [[] for i in range(n)]
	    for i in range(fps*time_steps):
	        for j, model in enumerate(agents):
	            st.session_state.vec_env[j] = model.get_env()
	            action, _states = model.predict(st.session_state.obs[j], deterministic=True)
	            st.session_state.obs[j], st.session_state.rewards[j], \
	            st.session_state.dones[j], st.session_state.info[j] = \
	            st.session_state.vec_env[j].step(action)
	            img = st.session_state.vec_env[j].render("rgb_array")
	            st.session_state.frames[j].append(img)
	        if stop_button:
	            break
	st.write("Done. Press Display to view their actions.")

if display_button:
    for i in range(len(st.session_state.frames[0])):
        for j, col in enumerate(cols):
            with col:
                containers[j].image(st.session_state.frames[j][i], use_column_width=True)        
        time.sleep(1/fps)
        if stop_button:
	        break
