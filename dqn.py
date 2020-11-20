from network import *
import torch
import numpy as np

class DQN:
    def __init__(self):
        
        self.q_network = Network(input_dimension=5, output_dimension=3)
        
        self.target_network = Network(input_dimension=5, output_dimension=3)
        
        self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr=0.001)
        
        self.TD_delta = 0
        
        self.gamma = 0.9
        
    def update_target(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
        
    def train_q_network(self, transition):
        # Set all the gradients stored in the optimiser to zero.
        self.optimiser.zero_grad()
        # Calculate the loss for this transition.
        loss = self._calculate_loss(transition)
        # Compute the gradients based on this loss, i.e. the gradients of the loss with respect to the Q-network parameters.
        loss.backward()
        # Take one gradient step to update the Q-network.
        self.optimiser.step()
        # Return the loss as a scalar
        return loss.item()
        
    def _calculate_loss(self, batch):
        states = np.array([batch[i][0] for i in range(len(batch))])
        action = np.array([batch[i][1] for i in range(len(batch))])
        reward = np.array([batch[i][2] for i in range(len(batch))])
        next_states = np.array([batch[i][3] for i in range(len(batch))])

        states = torch.tensor(states)
        action = torch.tensor(action)
        reward = torch.tensor(reward)
        next_states = torch.tensor(next_states)

        reward = torch.unsqueeze(reward.float(),1)
        action = torch.unsqueeze(action.long(), 1)

        network_prediction = self.q_network.forward(states)

        Q_values = torch.gather(network_prediction, 1, action)

        Q_target = self.target_network.forward(next_states).detach()

        best_target_action = torch.argmax(Q_target, dim=1)
        Q_target_state = torch.max(Q_target, dim=1)[0]

        #### Compute Weight loss here ####
        self.TD_delta = (torch.unsqueeze(Q_target_state,1) - Q_values)
        self.TD_delta = torch.flatten(self.TD_delta).tolist()

        bellman_loss = torch.nn.MSELoss()(reward + torch.unsqueeze(Q_target_state, 1) * self.gamma, Q_values)
    
        return bellman_loss