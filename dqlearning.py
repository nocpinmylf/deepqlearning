import gym
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Tạo Class Agent
class Agent:
  # Initialize
  def __init__(self,
              state_size_, 
              action_size_, 
              hidden_dim_=128,
              batch_size_ = 32,
              learning_rate_=0.001, 
              gamma_=0.99):
    self.state_size = state_size_
    self.action_size = action_size_
    
    # Khởi tạo các params
    self.replay_buffer = []
    self.gamma = gamma_
    self.hidden_dim = hidden_dim_
    self.batch_size = batch_size_
    self.epsilon = 1.0
    self.epsilon_min = 0.01
    self.epsilon_decay = 0.995 # (%)
    self.learning_rate = learning_rate_
    self.update_target_nn = 10
    
    # Build model
    self.main_network = self.getNeuralNetwork()
    self.target_network = self.getNeuralNetwork()
    
    # Update weight của target network
    self.updateTargetModel()
  
  # Tạo mạng neural
  def getNeuralNetwork(self):
    model = Sequential()
    model.add(Dense(self.hidden_dim, activation='relu', input_shape=(self.state_size,)))
    model.add(Dense(self.hidden_dim / 2, activation='relu'))
    model.add(Dense(self.action_size, activation='linear'))
    
    model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
    return model

  # lưu kinh nghiệm
  def saveExp(self, state, action, reward, next_state, terminated):
    self.replay_buffer.append((state, action, reward, next_state, terminated))
  
  # lấy batch để train main model
  def getBatchFromBuffer(self):
    exp_batch = random.sample(self.replay_buffer, self.batch_size)
    # lấy ra 5 mảng
    state_batch, action_batch, reward_batch, next_state_batch, terminated_batch = zip(*exp_batch)
    return state_batch, action_batch, reward_batch, next_state_batch, terminated_batch
  
  def trainMainModel(self):
    # lấy batch data
    state_batch, action_batch, reward_batch, next_state_batch, terminated_batch = self.getBatchFromBuffer()
    state_batch = np.vstack(state_batch)
    next_state_batch = np.vstack(next_state_batch)
    
    # Lấy Q value của state hiện tại
    q_value = self.main_network.predict(state_batch, verbose=0)
    next_q_value = self.target_network.predict(next_state_batch, verbose=0)
  
    # Tính toán Q value
    for i in range(self.batch_size):
      q_value[i][action_batch[i]] = reward_batch[i] if terminated_batch[i] else reward_batch[i] + self.gamma * np.max(next_q_value[i])
    
    # Huấn luyện
    self.main_network.fit(state_batch, q_value, verbose=0)
    if self.epsilon > self.epsilon_min:
      self.epsilon *= self.epsilon_decay
    
  def updateTargetModel(self):
    self.target_network.set_weights(self.main_network.get_weights())
  
  # Chọn action dựa trên epsilon-greedy policy
  def selectAction(self, state):
    if random.uniform(0, 1) < self.epsilon:
      # Chọn 1 hành động ngẫu nhiên
      return random.randrange(self.action_size)
    # Nếu epsilon nhỏ hơn sẽ tận dụng q max
    state = state.reshape((1, self.state_size))
    q_value = self.main_network.predict(state)
    return np.argmax(q_value[0])

if __name__ == "__main__":
  game_name = 'CartPole-v1'
  env = gym.make(game_name)
  state, info = env.reset(seed=42)
  
  state_dim = env.observation_space.shape[0]
  action_dim = env.action_space.n
  # print(env.observation_space)
  print(env.action_space.sample())
  episode = 20
  timestep = 500
  batch_size = 32
  
  total_timestep = 0
  # agent = Agent(state_size_=state_dim, action_size_=action_dim, batch_size_=batch_size)
  
  # for ep in range(episode):
  #   ep_reward = 0
  #   state, info = env.reset(seed=42)
  
  #   for t in range(timestep):
  #     total_timestep += 1
  
  #     # Update target network
  #     if total_timestep % agent.update_target_nn == 0:
  #       agent.updateTargetModel()
      
  #     # Lưu kinh nghiệm
  #     action = agent.selectAction(state)
  #     next_state, reward, terminated, truncated, info = env.step(action)
  #     agent.saveExp(state, action, reward, next_state, terminated)
      
  #     # Cập nhật exp, state
  #     state = next_state
  #     ep_reward += reward
      
  #     if terminated:
  #       print(f'Episode: {ep + 1} reached terminal with reward: {ep_reward}!')
  #       break
      
  #     # Train khi kho exp đủ batch size
  #     if len(agent.replay_buffer) > batch_size:
  #       agent.trainMainModel()
        
  # # Save weight (model)
  # agent.main_network.save('Trained Agent.h5')
      