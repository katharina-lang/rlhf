# Rlhf Project Documentation

- [Setup](#setup)
- [RLHF functionality](#rlhf-functionality)
- [Overview](#overview)
- [Data Collection](#data-collection)
- [Data Labeling](#data-labeling)
- [Reward Model Training](#reward-model-training)
- [Pretraining](#pretraining)
- [Client Server Structure](#client-server-structure)
- [Results](#results)
- [Additional Infos](#additional-infos)


## Setup

## RLHF Functionality

### Overview

The structure of our project can be seen in the file `rlhf/scripts/main.py`. The agent training has been taken from the CleanRL repository and their file `ppo_continous.py`. The core structure of their file has remained, but it is now modularized. In our project, we replaced the environment reward - which is used for the agent-and-critic's training - with a reward predictor.
The number of policy updates (num_iterations) is also the number of reward model updates. Data is collected in every iteration. Through ahe agent's interactions with the environment we receive observations as well as their corresponding actions, the environment rewards for each action and our predicted reward. Interaction with the environment is handled through the Singleton PPO which is instantiated in `main.py` and defined in `rlhf.core.ppo.py`.
The data is then labeled by either a human or a synthetic feedback supplier in order to receive feedback on the actions that were taken. As long as queries are still required, x pairs are labeled (x = max(3, 1+total_queries//num_iterations)). After the labeling process, the reward model is trained and then agent as well as critic are updated.


### Data Collection

Before the start of the training, we create three arrays, one for our observation-action pairs, the environment rewards and the predicted rewards respectively. Throughout the training, each training step adds a new index to each of these arrays so the indices for corresponding steps match between all three arrays. 
In order to start our data collection, we call the function `collect_rollout_data` from `ppo.py` in `main.py`.
After the environment interaction and reward prediction, `save_data` saves the collected data into the above-mentioned arrays.
We stack the data first and then reshape it at the end. The following example illustrates the data collection process: We have two environments and want to visualize the processes happening for the observation-action buffer. The dimension of the observation space is two and the action space dimension is one.
As we have two environments, two observation-action pairs are collected. Each index consists of the concatenated observation and its corresponding action, e.g. [1 2 3]. After the mentioned steps, the collected data looks as follows:

![obs-action-pair](/readme_images/obs_action/pairs_start.png)

For each step, the collected data is stacked with np.hstack and each row corresponds to all observation-action pairs in their correct order collected by one environment over the course of `collect_rollout_data`:

![obs-action-buffer](/readme_images/obs_action/pairs_stack.png)

After the agent-environment interaction is finished for the entire iteration, the data arrays are reshaped in such a way that the original observation-action pairs are reconstructed.

![obs-action-output](/readme_images/obs_action/pairs_output.png)

The same process applies to the predicted and environment rewards, with slight modifications. This ensures that each index now aligns across all arrays. The corresponding examples for the environment and predicted reward are provided in the [Data collection examples](#data-collection-examples) section.


### Data Labeling

After the data collection is completed, the collected data has to be labeled for feedback purposes. 
In every iteration, a labeling instance is created and its function `get_labeled_data` is called. This returns the labeled data in a format similar to the one described in the paper "Deep reinforcement learning from human preferences" by Christiano et al. (https://arxiv.org/pdf/1706.03741).
By calling `get_labeled_data()`, random segments are selected. A segment is a tuple of the trajectory and the the sum over all environment rewards for the corresponding observation-action pairs (env_reward).
We extract the amount of segments from the randomly selected segments that we want our human or synthetic feedback supplier to label during the labling process. Per default, an uncertainty-based method is used, but it is also possible to randomly select these pairs.
For the uncertainty-based approach, every reward model (there is an ensemble if wanted, more about this in [Reward Model Training](#reward-model-training)) predicts the reward for the trajectories. The sum is taken and the preference per model is saved. From these preferences, the variance is computed and saved. The labeled pairs with the highest variance are returned and we exit the labeling process and continue in `main.py`.
The return value is an array of triplets which each looks like the following: (trajectory1, trajectory2, (label1, label2)). A trajectory consists of n observation-action pairs (n = segment_size).


### Reward Model Training

The reward model is a simple feedforward neural network with 4 fully connected layers. The input dimension is the sum of the observation and action space dimensions `(obs_dim + action_dim)`. Each of the first three layers has 64 hidden units and uses the ReLU activation function. Additionally, dropout with a (default) rate of 30% is applied after each of these layers to prevent overfitting. The final layer outputs a single scalar value, representing the predicted reward, which is passed through a Tanh activation function to constrain the output to the range [-1, 1] which corresponds to a normalized reward.

The `train_reward_model_ensemble` function (called in `main.py`) is responsible for training an ensemble of reward models using labeled data. Each model is trained independently with its corresponding optimizer. The function supports mini-batch training, validation, and logging with TensorBoard. First, the function shuffles the labeled_data at the beginning of each epoch to ensure stochasticity in training. Then, the data is divided into mini-batches of size `batch_size_rm`. Each reward model in the ensemble is trained independently. To further increase training diversity, the order of the mini-batches is shuffled for each model. During the training process, for each mini-batch, the labeled observation-action pairs are passed through the model in a forward pass, where both segments in each pair are evaluated to predict their respective rewards. The cross-entropy loss is then computed by comparing the predicted probabilities with the true labels. This loss is backpropagated, and the optimizer updates the model’s parameters to minimize the loss.
If validation data (`val_data`) is provided, the function computes the validation loss for each model at the end of every epoch using the compute_reward_model_loss function. We do this to detect potential overfitting. The training and validation losses are logged using a TensorBoard writer, enabling the visualization of the training process and facilitating debugging or optimization. Once the training process is complete, all updated reward models in the ensemble are used later.


### Pretraining

## Client Server Structure
## Results
## Additional Infos
### Data collection examples
For the following examples the number of environments is also two.
#### environment reward
The data gets collected and reshaped immediately, so we can use the same procedure as for the observation_action pairs.

![env-reward-start](/readme_images/env_reward/env_start.png)

Then the reshaped data gets stacked on top of each other.

![env-reward-stack](/readme_images/env_reward/env_stack.png)

At last, the reshape process flattens the array so that the rewards are concatenated between environments.

![env-reward-output](/readme_images/env_reward/env_output.png)

#### predicted reward
The difference here is, that the data is saved as a tensor instead of a numpy array. The rest is analogue again.

![pred-reward-start](/readme_images/pred_reward/pred_start.png)

![pred-reward-cat](/readme_images/pred_reward/pred_cat.png)

![pred-reward-output](/readme_images/pred_reward/pred_output.png)

