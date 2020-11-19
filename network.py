import torch
# The Network class inherits the torch.nn.Module class, which represents a neural network.
class Network(torch.nn.Module):

    # The class initialisation function. This takes as arguments the dimension of the network's input (i.e. the dimension of the state), and the dimension of the network's output (i.e. the dimension of the action).
    def __init__(self, input_dimension, output_dimension):
        # Call the initialisation function of the parent class.
        super(Network, self).__init__()
        
        # Config the network
#         self.config = network_config
        
        # Define the network layers. This example network has two hidden layers, each with 100 units.
        self.layer_1 = torch.nn.Linear(in_features=input_dimension, out_features=100)
        
        
        self.layer_2 = torch.nn.Linear(in_features=100, out_features=100)
        
        self.V_layer = torch.nn.Linear(in_features=100, out_features=1)
        
        self.a_layer = torch.nn.Linear(in_features=100, out_features=output_dimension)
        

    # Function which sends some input data through the network and returns the network's output. In this example, a ReLU activation function is used for both hidden layers, but the output layer has no activation function (it is just a linear layer).
    def forward(self, input):
        layer_1_output = torch.nn.functional.relu(self.layer_1(input.float()))
        layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))
        
        V = self.V_layer(layer_2_output)
        
        a = self.a_layer(layer_2_output)
        
        q = V + a - a.mean()
        
        return q

    
# import torch
# # The Network class inherits the torch.nn.Module class, which represents a neural network.
# class Network(torch.nn.Module):

#     # The class initialisation function. This takes as arguments the dimension of the network's input (i.e. the dimension of the state), and the dimension of the network's output (i.e. the dimension of the action).
#     def __init__(self, input_dimension, output_dimension):
#         # Call the initialisation function of the parent class.
#         super(Network, self).__init__()
        
#         # Config the network
# #         self.config = network_config
        
#         # Define the network layers. This example network has two hidden layers, each with 100 units.
        
#         self.rnn_market = torch.nn.GRU(10, 1, 2)
#         self.rnn_rsi = torch.nn.GRU(10, 1, 2)
        
#         self.h1 = torch.zeros(2,1,1)
        
#         self.h2 = torch.zeros(2,1,1)
        
        
#         self.layer_1 = torch.nn.Linear(in_features=4, out_features=100)
        
        
#         self.layer_2 = torch.nn.Linear(in_features=100, out_features=100)
        
#         self.V_layer = torch.nn.Linear(in_features=100, out_features=1)
        
#         self.a_layer = torch.nn.Linear(in_features=100, out_features=output_dimension)
        

#     # Function which sends some input data through the network and returns the network's output. In this example, a ReLU activation function is used for both hidden layers, but the output layer has no activation function (it is just a linear layer).
#     def forward(self, input):
        
#         T = torch.cat((torch.tensor(input[0]), 
#                torch.tensor(input[1]), 
#                torch.tensor([input[2]]),
#                torch.tensor([input[3]])
#               ))
        
#         O, self.h1 = self.rnn_market(T[:10].reshape(1,1,-1), self.h1)
        
        
#         rnn_market_output = torch.nn.functional.relu(O)
        
#         O, self.h2 = self.rnn_rsi(T[10:20].reshape(1,1,-1), self.h2)
        
#         rnn_rsi_output = torch.nn.functional.relu(O)
        
#         rnn_output = torch.cat((rnn_market_output.reshape(-1), rnn_rsi_output.reshape(-1), T[20:]))
        
#         layer_1_output = torch.nn.functional.relu(self.layer_1(rnn_output))
        
#         layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))
        
#         V = self.V_layer(layer_2_output)
        
#         a = self.a_layer(layer_2_output)
        
#         q = V + a - a.mean()
        
#         return q
