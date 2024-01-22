import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import gym
from gym.spaces import Discrete, Box
from utils import mlp, Net
import os




def reward_to_go(rews):
    n = len(rews)
    rtgs = np.zeros_like(rews)
    for i in reversed(range(n)):
        rtgs[i] = rews[i] + (rtgs[i+1] if i+1 < n else 0)
    return rtgs


#function to train a vanilla policy gradient agent. By default designed to work with Cartpole
def train(env_name='CartPole-v0', hidden_sizes=[32], lr=1e-2, 
          epochs=50, batch_size=5000, render=False, reward=None, checkpoint = False, checkpoint_dir = "\."):

    # make environment, check spaces, get obs / act dims
    env = gym.make(env_name)
    assert isinstance(env.observation_space, Box), \
        "This example only works for envs with continuous state spaces."
    assert isinstance(env.action_space, Discrete), \
        "This example only works for envs with discrete action spaces."

    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n

    # make core of policy network
    logits_net = mlp(sizes=[obs_dim]+hidden_sizes+[n_acts])

    # make function to compute action distribution
    def get_policy(obs):
        logits = logits_net(obs)
        return Categorical(logits=logits)

    # make action selection function (outputs int actions, sampled from policy)
    def get_action(obs):
        return get_policy(obs).sample().item()

    # make loss function whose gradient, for the right data, is policy gradient
    def compute_loss(obs, act, weights):
        logp = get_policy(obs).log_prob(act)
        return -(logp * weights).mean()

    # make optimizer
    optimizer = Adam(logits_net.parameters(), lr=lr)

    # for training policy
    def train_one_epoch():
        # make some empty lists for logging.
        batch_obs = []          # for observations
        batch_acts = []         # for actions
        batch_weights = []      # for reward-to-go weighting in policy gradient
        batch_rets = []         # for measuring episode returns
        batch_lens = []         # for measuring episode lengths

        # reset episode-specific variables
        obs = env.reset()       # first obs comes from starting distribution
        done = False            # signal from environment that episode is over
        ep_rews = []            # list for rewards accrued throughout ep

        # render first episode of each epoch
        finished_rendering_this_epoch = False

        # collect experience by acting in the environment with current policy
        while True:

            # rendering
            if (not finished_rendering_this_epoch) and render:
                env.render()

            # save obs
            batch_obs.append(obs.copy())

            # act in the environment
            act = get_action(torch.as_tensor(obs, dtype=torch.float32))
            obs, rew, done, _ = env.step(act)

            if reward is not None:
            #replace reward with predicted reward from neural net
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                torchified_state = torch.from_numpy(obs).float().to(device)
                r = reward.predict_return(torchified_state.unsqueeze(0)).item()
                rew = r

            # save action, reward
            batch_acts.append(act)
            ep_rews.append(rew)

            if done:
                # if episode is over, record info about episode
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # the weight for each logprob(a_t|s_t) is reward-to-go from t
                batch_weights += list(reward_to_go(ep_rews))

                # reset episode-specific variables
                obs, done, ep_rews = env.reset(), False, []

                # won't render again this epoch
                finished_rendering_this_epoch = True

                # end experience loop if we have enough of it
                if len(batch_obs) > batch_size:
                    break

        # take a single policy gradient update step
        optimizer.zero_grad()
        batch_loss = compute_loss(obs=torch.as_tensor(batch_obs, dtype=torch.float32),
                                  act=torch.as_tensor(batch_acts, dtype=torch.int32),
                                  weights=torch.as_tensor(batch_weights, dtype=torch.float32)
                                  )
        batch_loss.backward()
        optimizer.step()
        return batch_loss, batch_rets, batch_lens

    # training loop
    for i in range(epochs):
        batch_loss, batch_rets, batch_lens = train_one_epoch()
        if reward is not None:
            print('epoch: %3d \t loss: %.3f \t predicted return: %.3f \t ep_len (gt reward): %.3f'%
                (i, batch_loss, np.mean(batch_rets), np.mean(batch_lens)))
        else:
            print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f'%
                (i, batch_loss, np.mean(batch_rets), np.mean(batch_lens)))
            
        if checkpoint:
            #checkpoint after each epoch
            print("!!!!!! checkpointing policy !!!!!!")
            torch.save(logits_net.state_dict(), checkpoint_dir + '/policy_checkpoint'+str(i)+'.params')
    
    #always at least checkpoint at end of training
    if not checkpoint:
        torch.save(logits_net.state_dict(), checkpoint_dir + '/final_policy.params')
    

if __name__ == '__main__':
    print("testing")
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str, default='CartPole-v0')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--checkpoint', action='store_true')
    parser.add_argument('--checkpoint_dir', type=str, default='\.')
    parser.add_argument('--reward_params', type=str, default='', help="parameters of learned reward function")
    args = parser.parse_args()
    
    
    #create checkpoint directory if it doesn't already exist
    isExist = os.path.exists(args.checkpoint_dir)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(args.checkpoint_dir)

    if args.reward_params == '':
        #train on ground-truth reward function
        train(env_name=args.env_name, render=args.render, lr=args.lr, 
              epochs=args.epochs, checkpoint=args.checkpoint, 
              checkpoint_dir=args.checkpoint_dir)
    else:
        #pass in parameters for trained reward network and train using that
        print("training on learned reward function")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        reward_net = Net()
        reward_net.load_state_dict(torch.load(args.reward_params))
        reward_net.to(device)
        train(env_name=args.env_name, render=args.render, lr=args.lr, 
              epochs=args.epochs, reward=reward_net, checkpoint=args.checkpoint, 
              checkpoint_dir=args.checkpoint_dir)