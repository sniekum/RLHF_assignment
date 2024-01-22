import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import gym
from gym.spaces import Discrete, Box
import matplotlib.pyplot as plt
from vpg import mlp




#execute the policy to generate one rollout and keep track of performance
def generate_rollout(policy, env, rendering = False):

    # make action selection function (outputs int action, sampled from policy)
    def get_action(policy, obs):
        logits = policy(obs)
        return Categorical(logits=logits).sample().item()


    # reset episode-specific variables
    obs = env.reset()       # first obs comes from starting distribution
    done = False            # signal from environment that episode is over
    ep_rews = []            # list for rewards accrued throughout ep

    cum_ret = 0
    obs_traj = []
    # collect experience by acting in the environment with current policy
    while not done:

        # rendering
        if rendering:
            env.render()
        # act in the environment
        act = get_action(policy, torch.as_tensor(obs, dtype=torch.float32))
        obs, rew, done, _ = env.step(act)
        cum_ret += rew
        obs_traj.append(obs)



    return obs_traj, cum_ret
        
        

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str, default='CartPole-v0')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--checkpoint', type=str, default='', help="pretrained policy weights")
    parser.add_argument('--num_rollouts', type=int, default=1)
    
    args = parser.parse_args()

    checkpoint = args.checkpoint

    # make core of policy network
    env = gym.make(args.env_name)
    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n
    hidden_sizes=[32]
    policy = mlp(sizes=[obs_dim]+hidden_sizes+[n_acts])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    policy.load_state_dict(torch.load(checkpoint))
    
    returns = 0
    for i in range(args.num_rollouts):
        _, cum_ret = generate_rollout(policy, env, rendering=args.render)
        print("cumulative return", cum_ret)
        returns += cum_ret
    print("average return", returns/args.num_rollouts)