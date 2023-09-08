import gym
import random
import numpy as np
import tensorflow as tf
  
game_name = 'CartPole-v1'
env = gym.make(game_name, render_mode='human')
state, info = env.reset(seed=42)
state_size = env.observation_space.shape[0]

# load model
model_path = 'Trained Agent.h5' 
agent = tf.keras.models.load_model(model_path)

timestep = 500
total_rewards = 0

for i in range(timestep):
  env.render()
  state = state.reshape((1, state_size))
  q_values = agent.predict(state)
  
  observation, reward, terminated, truncated, info = env.step(np.argmax(q_values))
  total_rewards += reward
  state = observation