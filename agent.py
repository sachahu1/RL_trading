# State : stock price, RSI, number of shares, reward
# reward can be wallet
import numpy as np
import torch
from dqn import *
from replaybuffer import *

class Agent: # Decision making, RL agent
    def __init__(self, agent_config, buffer_config):
        
        self.agent_config = agent_config
        
        self.buffer_config = buffer_config 
        
        self.state = None
        
        self.action = None
        
        self.epsilon = 1
        
        self.delta = agent_config["delta"]
        
        self.dqn = DQN()
        
        self.Buffer = ReplayBuffer(**self.buffer_config)
        
        
    def step(self, action = None):

        actions = [0, 1, 2] # Buy , Sell , Hold 

        p = np.ones(3)*((self.epsilon)/3)

        p[action] = 1-self.epsilon + self.epsilon/3
        discrete_action = np.random.choice(actions, p=p)

        if self.epsilon - self.delta > self.agent_config["epsilon_clip"]:
            self.epsilon = self.epsilon - self.delta
        else : 
            self.epsilon = self.agent_config["epsilon_clip"]
        
        return discrete_action
        
    # Function to get the next action, using whatever method you like
    def get_next_action(self, state):
        # Here, the action is random, but you can change this
        Q_values = self.dqn.q_network.forward(torch.tensor(state).float()).detach().numpy()
        action = np.argmax(Q_values)
        
        action = self.step(action=action)
        
        # Store the state; this will be used later, when storing the transition
        self.state = state
        # Store the action; this will be used later, when storing the transition
        self.action = action

        return action
    
    # Function to set the next state and distance, which resulted from applying action self.action at state self.state
    def set_next_state(self, next_state, reward):
        # Create reward function
#         reward = next_state[3] - self.state[3]
        
        # Create a transition
        transition = (self.state, self.action, reward, next_state)
        # Save transition
        self.Buffer.save_transition(transition)
        # If the buffer contains enough data to sample a batch
        if (len(self.Buffer.buf)>=self.Buffer.batch_size):
            batch, chosen_indexes = self.Buffer.sample_batch()
            
            loss = self.dqn.train_q_network(batch)
            self.Buffer.update_weight(chosen_indexes, self.dqn.TD_delta)

                


