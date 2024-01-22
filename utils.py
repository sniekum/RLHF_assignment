import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np


# Helper function to construct a feedforward multilayer perception that you can use in class Net if you want.
# Do NOT modify, as this is used to construct the policy network.
# Sizes is a list of the number of neurons in each layer (where the number of layers is the len of the list)
# nn.Sequential allows you to skip explicitly defining a forward pass like you had to do in assignment 1. 
# See pyTorch documentation if you're confused (or feel free to define the network and forward() by hand as in assignment 1).
def mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity):
    # Build a feedforward neural network for the policy
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


class Net(nn.Module):
    ##TODO Define the reward network architecture and functionality
    # Be sure to implement both __init__() and predict_reward()

    def __init__(self):
        super().__init__()
        #TODO define network architecture. 
        # Feel free to use the mlp helper function above or construct the network and forward() manually like you did in assignment 1.
        # Hint: states in cartpole are 4-dimensional (x,xdot,theta,thetadot)
        # https://www.gymlibrary.dev/environments/classic_control/cart_pole/
   

    def predict_return(self, traj):
        '''calculate return (cumulative reward) of a trajectory (could be any number of timesteps)'''
        #TODO should take in a trajectory and output a scalar cumulative reward estimate
        




	

   