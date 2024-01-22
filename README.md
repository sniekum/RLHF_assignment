# RLHF_assignment

You will be using the conda environment that you created in assignment 1. After cloning this repository, navigate to it and then run:
```
conda activate imitation_learning
```

## Assignment

You will need fill in code for a neural network reward function in ```utils.py``` and write the code to train it ```offline_reward_learning.py```. I recommend looking back at HW1 and the Pytorch tutorial if you're unsure how to start.

## Part 1: Synthetic demonstration generation

We will use a vanilla policy gradient algorithm to create synthetic preference data. To understand the RL code in ```vpg.py``` you may want to review the Introduction to RL (parts 1-3) here: https://spinningup.openai.com/en/latest/spinningup/rl_intro.html#
The reward-to-go policy gradient in ```vpg.py``` is very simple and adapted from the above introduction. PPO, which is more commonly used (but also more complex to work with), is just a slightly fancier policy gradient algorithm. 

Now look through the code in ```vpg.py``` to get an idea of what is going on. We will use RL to generate some suboptimal trajectories and we will use the ground-truth reward to provide synthetic preference labels.

Run 
```
python vpg.py --epochs 10 --checkpoint --checkpoint_dir ./synthetic --render
```

This will generate 10 and save the RL policy after 10 checkpoints on the CartPole environment. It will render the first episode of each epoch so you can see its performance visually. We won't train the RL policy to convergence, but if you ran it for more than 10 checkpoints it would eventually reach a return of 200 (the ground truth cartpole reward function is +1 every time step that the cart stays on the track and the pole stays roughly upright). You can learn more about CartPole here: https://www.gymlibrary.dev/environments/classic_control/cart_pole/


## Part 2: Create pairwise preferences and train a reward function network


Open ```utils.py``` and  ```offline_reward_learning.py```. 
In ```utils.py``` you will need to code up a reward function network. CartPole is pretty simple so it doesn't need to be very complicated. A 2-layer feed forward network is sufficient. There are a few hints in the comments about what to do.

In ```offline_reward_learning.py``` you will use the saved checkpoints from Part 1 to generate synthetic preferences over suboptimal trajectories. The data generation is done for you in ```generate_novice_demos``` and ```create_training_data```. If you look at the main method you will see that you need to implement ```learn_reward``` to train your reward network using pairwise preferences. As in the paper Extrapolating Beyond Suboptimal Demonstrations via Inverse Reinforcement Learning from Observations, which we read in class, you should use a CrossEntropyLoss where the logits are the cumulative predicted rewards for each trajectory in the pair and the classification label is whichever trajectory should be preferred based on ground-truth return. Again, there are hints in the comments.

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

Let's now evaluate the learned RLHF policy and compare with the performance of the final checkpoint policy that we used to create trajectories:

To estimate the expected return (via rollouts) of your RLHF policy, run the command below and fill in the appropriate final checkpoint number for XX (49 if you ran for 50 epochs)
```
python rollout_policy.py --checkpoint ./rlhf/policy_checkpointXX.params --num_rollouts 5 --render
```

Report average performance.

Now, evaluate final checkpoint that was used to generate the trajectories for the training preference dataset:
```
python rollout_policy.py --checkpoint ./synthetic/policy_checkpoint9.params --num_rollouts 5 --render
```

Report the average performance.

Finally, edit rollout_policy.py and use matplotlib to generate a line graph of the average performance across all 50 checkpoints for your RL policy and include it in your writeup.

Now answer the following questions: (1) Was your RLHF method able to learn a better policy than the best policy used to collect training data? (2) Why is it possible to learn a policy much better than the best trajectory in the training data? (3) Explain how RLHF is different from Inverse RL or behavioral cloning. (4) What would have happened if you ran Inverse RL or behavioral cloning on the same trajectories used for RLHF?

## Submission.

Prepare a PDF report (preferably in LaTeX) with your answers to the questions and submit the PDF along with your code in a zip file on Gradescope.

### Acknowledgement
Based on an assignment originally created by Daniel S. Brown. 

