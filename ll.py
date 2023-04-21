import streamlit as st
from model_utils import train_model
import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
import os
import imageio

def frames_to_video(frames, fps, filename):
    """
    Convert a list of frames to a video file.
    """
    with imageio.get_writer(filename, fps=fps) as writer:
        for frame in frames:
            writer.append_data(frame)
    return filename

class trainedModel():
	def __init__(self,name,model):
		self.name = name
		self.model = model

def load_options(env):
	model_options = [trainedModel('Random Model',DQN(
                "MlpPolicy",
                "LunarLander-v2",
                verbose=1,
                exploration_final_eps=0.1,
                target_update_interval=250))]

	for path in os.listdir('models/'):
		model = DQN.load('models/'+path[:-4],env=env)
		model_options.append(trainedModel(path[:-4],model))

	return model_options


def main():

	env = gym.make("LunarLander-v2", render_mode='rgb_array')
	if not os.path.exists('models/'):
		os.mkdir('models')

	# sidebar for training and selecting models

	if 'options' not in st.session_state:
		st.session_state.options = []
	with st.sidebar:
		st.info('Click on refresh models to update model list')
		col1, col2 = st.columns(2)
		with col1:	
			if st.button('Refresh Models'):	
				st.session_state.options = load_options(env)
		with col2:
			if st.button('Delete Models'):
				folder_path = 'models/'
				for file_name in os.listdir(folder_path):
				    if file_name != 'Best model.zip':
				        file_path = os.path.join(folder_path, file_name)
				        try:
				            os.remove(file_path)
				            print(f"Deleted {file_name}")
				        except Exception as e:
				            print(f"Error deleting {file_name}: {e}")		
				st.session_state.options = load_options(env)            
		models = st.multiselect('Select models',st.session_state.options,format_func=lambda x: x.name, max_selections = 4)

		start_prob = st.slider('Select initial value of random action probability (recommended = 1)', min_value=0.0, max_value=1.0, step=0.01, value=1.0)
		final_prob = st.slider('Select final value of random action probability (recommended = 0.1)', min_value=0.0, max_value=1.0, step=0.01, value=0.1)
		time_steps = st.slider('Select no. of timesteps to train the model (recommended = 100000)', min_value=0, max_value=500_000, step=1000, value=100_000)
		lr = st.slider('Select learning rate (recommended = 0.0001)', min_value=0.0, max_value=0.1, step=0.0001, value=0.0001,format='%f')
		name = st.text_input('Name your model')
		if st.button('Train model'):
			with st.spinner('Model is under training'):
				model = train_model(time_steps,start_prob,final_prob,lr)
				model.save(f"models/{name}")
			st.write('Model trained successfully')	
		st.info("Note: Training may take some time.")	

	# sidebar end

	n = len(models) if len(models) else 1
	if 'vec_env' not in st.session_state or len(st.session_state.vec_env)!=n:
		st.session_state.vec_env = [None for i in range(n)]
	if 'obs' not in st.session_state or len(st.session_state.obs)!=n:
	    st.session_state.obs = [None for i in range(n)]
	if 'rewards' not in st.session_state or len(st.session_state.rewards)!=n:
	    st.session_state.rewards = [None for i in range(n)]
	if 'dones' not in st.session_state or len(st.session_state.dones)!=n:
	    st.session_state.dones = [None for i in range(n)]
	if 'info' not in st.session_state or len(st.session_state.info)!=n:
	    st.session_state.info = [None for i in range(n)]
	if 'frames' not in st.session_state or len(st.session_state.frames)!=n:
	    st.session_state.frames = [[] for i in range(n)]
	st.title('Lunar Lander')
	st.write('#### To Train your model, access the sidebar')
	st.info('This model uses Deep Q Network learning technique. For more info scroll down')


	st.write('#### Generate a video of bot playing') 


	col1, col2, col3, col4, col5 = st.columns(5)
	with col1:
		start_button = st.button(label='Generate')
	with col2:
		display_button = st.button(label='Display video')  

	cols = st.columns(n)
	containers = [st.empty() for i in range(n)]
	for i, col in enumerate(cols):
	    with col:
	        containers[i] = st.empty()
	       
	fps = st.number_input('FPS', min_value=10, max_value=240, value=60, step=5)
	time_sec = st.number_input('How long should bot play?', min_value = 10, max_value = 60, value = 10, step=5)

	if start_button:
		with st.spinner("Bots are playing..."):
		    for i, model in enumerate(models):
		        st.session_state.vec_env[i] = model.model.get_env()
		        st.session_state.obs[i] = st.session_state.vec_env[i].reset()
		        st.session_state.frames = [[] for i in range(n)]
		    # for i in range(fps*time_steps):
		    #     for j, model in enumerate(models):
		    #         st.session_state.vec_env[j] = model.model.get_env()
		    #         action, _states = model.model.predict(st.session_state.obs[j], deterministic=True)
		    #         st.session_state.obs[j], st.session_state.rewards[j], \
		    #         st.session_state.dones[j], st.session_state.info[j] = \
		    #         st.session_state.vec_env[j].step(action)
		    #         img = st.session_state.vec_env[j].render("rgb_array")
		    #         st.session_state.frames[j].append(img)
		    for i, model in enumerate(models):
		    	for j in range(fps*time_sec):
		    		st.session_state.vec_env[i] = model.model.get_env()
		    		action, _states = model.model.predict(st.session_state.obs[i], deterministic=True)
		    		st.session_state.obs[i], st.session_state.rewards[i], \
		    		st.session_state.dones[i], st.session_state.info[i] = \
		    		st.session_state.vec_env[i].step(action)
		    		img = st.session_state.vec_env[i].render("rgb_array")
		    		st.session_state.frames[i].append(img)
		st.write("Done. Press Display to view their actions.")

	if display_button:
		for i, frames in enumerate(st.session_state.frames):
			frames_to_video(frames,fps,f'{models[i].name}.mp4')
		for j, col in enumerate(cols):
			with col:
				st.write(models[j].name)
				video_file = open(f'{models[j].name}.mp4', 'rb')
				video_bytes = video_file.read()
				containers[j].video(data=video_bytes, start_time=0) 

	st.info("""### About DQN
	DQN is a machine learning algorithm that uses a neural network to estimate the value of 
	different actions in a given state.The algorithm balances exploration and exploitation 
	by using an exploration probability.
	The model needs to explore to look for better solution but it needs to exploit to use the knowledge it has
	- Exploration probability starts high and decreases gradually.")
	- The learning rate controls how much the internal values are updated after each action.""")
    
if __name__	 == '__main__':
	st.set_page_config(layout='wide')

	main()