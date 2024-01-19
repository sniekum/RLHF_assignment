# RLHF_assignment

You will be using the conda environment that you created in assignment 1. After cloning this repository, navigate to it and then run:
```
conda activate imitation_learning
```

## Assignment

You will need fill in code for a neural network reward function in ```utils.py``` and write the code to train it ```offline_reward_learning.py```. I recommend looking back at HW1 and the Pytorch tutorial if you're unsure how to start.

## Part 1: Synthetic demonstration generation

We will use a vanilla policy gradient algorithm to create synthetic preference data. To understand the RL code in ```vpg.py``` read through the Introduction to RL (parts 1-3) here: https://spinningup.openai.com/en/latest/spinningup/rl_intro.html#
This will give you a introduction to policy gradient algorithms. There are many great RL libraries with much better RL implementations, but they are often a pain to set up so we will use the barebones implementation in ```vpg.py``` which is adapted from the above introduction and uses the reward-to-go policy gradient. PPO (which we won't use) is just a fancier policy gradient algorithm. You can optionally read about it here: https://spinningup.openai.com/en/latest/algorithms/ppo.html if you're interested in more details.

Now look through the code in ```vpg.py``` to get an idea of what is going on. We will use RL to generate some suboptimal trajectories and we will use the ground-truth reward to provide synthetic preference labels.

Run 
```
python vpg.py --epochs 10 --checkpoint --checkpoint_dir ./synthetic --render
```

This will generate 10 and save the RL policy after 10 checkpoints on the CartPole environment. It will render the first episode of each epoch so you can see its performance visually. We won't train the RL to convergence, but if you ran it for more than 10 checkpoints it would reach a reward of 200 (the true cartpole reward is +1 every time step that the cart stays on the track and the pole stays roughly upright). You can learn more about CartPole here: https://www.gymlibrary.dev/environments/classic_control/cart_pole/


## Part 2: Create pairwise preferences and train a reward function network


Open ```utils.py``` and  ```offline_reward_learning.py```. 
In ```utils.py``` you will need to code up a reward function network. CartPole is pretty simple so it doesn't need to be very complicated. Probably a 2-layer network is sufficient. 

In ```offline_reward_learning.py``` you will use the saved checkpoints from Part 1 to generate synthetic preferences over suboptimal trajectories. The data generation is done for you in ```generate_novice_demos``` and ```create_training_data```. If you look at the main method you will see that you need to implement ```learn_reward``` to train your network using pairwise preferences. As talked about in class, you can use a CrossEntropyLoss where the logits are the cumulative predicted rewards for each trajectory in the pair and the classification label is whatever trajectory should be more preferred.

Train a reward function using preferences over trajectories labeled using ground-truth rewards

```
python .\offline_reward_learning.py
```

This should save the learned reward function weights in a file called ```reward.params```.



## Part 3: Run RL on the learned reward function
Train RL policy on learned reward function

```
python vpg.py --epochs 50 --checkpoint --reward reward.params --checkpoint_dir rlhf
```

This will run for 50 epochs and checkpoint after each epoch. You may need to run a little longer, but you should be able to get close to episode lengths of 200 which corresonds to optimal performance for this task.

Let's now evaluate the learned RLHF policy and compare with the performance of the best checkpoint policy that we used to create trajectories:

Fill in appropriate checkpoint numbers (49 if you ran for 50 epochs)
```
python rollout_policy.py --checkpoint ./rlhf/policy_checkpointXX.params --num_rollouts 5 --render
```

Report average performance.

Evaluate best training checkpoint:
```
python rollout_policy.py --checkpoint ./synthetic/policy_checkpoint9.params --num_rollouts 5 --render
```

Report the average performance.

Was your RLHF method able to learn a better policy? Why is it possible to learn a policy much better than the best trajectory in the training data?
Explain how RLHF is different from Inverse RL or behavioral cloning. What would have happened if you ran Inverse RL or behavioral cloning on the same trajectories used for RLHF?

## Submission.

Prepare a pdf report with your answers to the questions and submit the pdf along with your code in a zip file. Submit via Canvas.



