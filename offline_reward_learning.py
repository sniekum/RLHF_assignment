import gym
import pygame
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
import random
from rollout_policy import generate_rollout
from utils import mlp, Net


def generate_novice_demos(env):
    checkpoints = []
    for i in range(10):
        checkpoints.append("./synthetic/policy_checkpoint"+str(i)+".params")

    # make core of policy network
    env = gym.make("CartPole-v0")
    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n
    hidden_sizes=[32]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    


    demonstrations = []
    demo_returns = []
    
    for checkpoint in checkpoints:

        policy = mlp(sizes=[obs_dim]+hidden_sizes+[n_acts],activation=nn.Tanh)
        policy.load_state_dict(torch.load(checkpoint))
        traj, ret = generate_rollout(policy, env)
        print("traj ground-truth return", ret)
        demonstrations.append(traj)
        demo_returns.append(ret)
         
    return demonstrations, demo_returns


def create_training_data(trajectories, cum_returns, num_pairs):
    training_pairs = []
    training_labels = []
    num_trajs = len(trajectories)

    #construct pairwise preferences over full trajectories
    for n in range(num_pairs):
        ti = 0
        tj = 0
        #pick two different trajectories 
        while(ti == tj):
            ti = np.random.randint(num_trajs)
            tj = np.random.randint(num_trajs)
        traj_i = trajectories[ti]
        traj_j = trajectories[tj]
        
	# Label based on cumulative returns 
        if cum_returns[ti] > cum_returns[tj]:
            label = 0
        else:
            label = 1
        # print(cum_returns[ti], cum_returns[tj])
        # print(label)
        
        training_pairs.append((traj_i, traj_j))
        training_labels.append(label)
       
    return training_pairs, training_labels


# NOTE: This is a function for debugging. Do not use it to compute returns when you implement learn_reward(). 
# Instead, use Net.predict_return() so that you get back a tensor that does the bookeeping for autograd
def predict_traj_return(net, traj):
    traj = np.array(traj)
    traj = torch.from_numpy(traj).float().to(device)
    return net.predict_return(traj).item()
    

# Train the network
def learn_reward(reward_network, optimizer, training_inputs, training_outputs, num_iter, checkpoint_dir):
    #check if gpu available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    #We will use a cross entropy loss for pairwise preference learning
    loss_criterion = nn.CrossEntropyLoss()
    
    #TODO: train reward function using the training data for the specified number of epochs (num_iter).
    # training_inputs gives you a list of pairs of trajectories
    # training_outputs gives you a list of labels (0 if first trajectory better, 1 if second is better)
    # Hint: For simplicity, don't try to batch multiple preferences; just compute loss and update parameters for each preference separately
    # Hint: Check the pytorch documentation for details on the tensor dimensions it expects
    # Hint: You may find torch.cat useful.
    # Hint: You may also find unsqueeze useful. For example, if x is the tensor [1,2,3] (which has shape [3]), then x.unsqueeze(0) returns [[1,2,3]] (which has shape [1,3])




    #After training we save the reward function weights    
    print("checkpointing")
    torch.save(reward_net.state_dict(), checkpoint_dir)
    print("finished training")


if __name__=="__main__":

    env = gym.make("CartPole-v0")
    
    num_pairs = 20
    #create synthetic trajectories for RLHF
    trajectories, traj_returns = generate_novice_demos(env)

    #create pairwise preference data using ground-truth reward
    traj_pairs, traj_labels = create_training_data(trajectories, traj_returns, num_pairs)
    
    #TODO: hyper parameters that you may want to tweak or change
    num_iter = 100
    lr = 0.001
    checkpoint = "./reward.params" #where to save your reward function weights

    # Now we create a reward network and optimize it using the training data.
    #TODO: You will need to code up Net in utils.py   
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    reward_net = Net()
    reward_net.to(device)


    import torch.optim as optim
    optimizer = optim.Adam(reward_net.parameters(),  lr=lr)
    #TODO: You will need to implement learn_reward, you can add arguments or do whatever you want
    learn_reward(reward_net, optimizer, traj_pairs, traj_labels, num_iter, checkpoint)


    #debugging printout
    #we should see higher predicted rewards for more preferred trajectories
    print("performance on training data")
    for i,pair in enumerate(traj_pairs):
        trajA, trajB = pair
        print("predicted return trajA", predict_traj_return(reward_net, trajA))
        print("predicted return trajB", predict_traj_return(reward_net, trajB))
        if traj_labels[i] == 0:
            print("A should be better\n")
        else:
            print("B should be better\n")

