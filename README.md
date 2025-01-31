# Rlhf Project Documentation

- [Setup](Setup)
- [RLHF functionality](RLHF-Functionality)
- [Overview](Overview)
- [Data Collection](Data-Collection)
- [Data Labeling](Data-Labeling)
- [Reward Model Training](Reward-Model-Training)
- [Client Server Structure](Client-Server-Structure)
- [Results](Results)
- [Additional Infos](Additional-Infos)


## Setup

## RLHF Functionality

### Overview

The structure of our Project can be seen in the file `rlhf/scripts/main.py`. The agent training has been taken from the CleanRL repository and their file `ppo_continous.py`. The core structure from their file remains but it is modularized. In our project we replaced the environment reward which is used for the agents (and critics) training with a reward predictor.
The number of policy updates (num_iterations) also is the number of reward model updates. Every iteration, data is collected. The agent interacts with the environment and through this we get observations with their corresponding actions, the environemt rewards for an action and our predicted reward. Interaction with the environment is handeled through the Singleton PPO which is instanciated in `main.py` and defined in `rlhf.core.ppo.py`.
Then we label the data. As long as we still want queries, x pairs get labeled (x = max(3, 1+total_queries//num_iterations)). After the labeling process the reward model is trained and then agent and critic are updated.

### Data Collection

Per data category (observation action pair,environment_reward, predicted_reward) we want a flat array. From the file `ppo.py` the function `collect_rollout_data` gets called in `main.py`.
After the environment interaction and reward prediction `save_data` saves the data into arrays.
The data which is collected looks like the following:
# Image
When new data is collected, the data is stacked with np.hstack:
# Image
After the agent environment interaction is finished for the whole iteration the every data array gets flattened.
# Image
Like this we ensure that each indice corresponds to the related data.
This is analog for the predicted and environment reward.
A single observation action pair is a flat array itself with obs_dim+action_dim elements. The rewards are just numbers. For consitency these numbers are also saved in a array with the reward value as the element.



### Data Labeling
Every iteration a Labeling instance is created and it's function `get_labeled_data` is called. This returns the labeled data in a format similar to the one in the paper "Deep reinforcement learning from human preferences".
A array of triplet is returned here. One triplet looks like the following (trajectory1, trajectory2, (label1, label2)) and a trajectory consists of n observation action pairs (n=segment_size).
Through `get_labeled_data()` random segments get selected and a segment is a triplet of the trajectory, the env_reward and the predicted reward. The environment reward is the sum over all env_rewards for the corresponding observation action pairs.
# evtl tuple wenn wir predicted reward entfernen
After segment selection the wanted number of labeled pairs are created. Per default a uncertainty based method is used but it is also possible to randomly select these pairs.
For the uncertainty based approach, every reward model (there is an ensemble if wanted, more about this in [Reward Model Training](Reward-Model-Training)) predicts the reward for the trajectories. The sum is taken and the preference per model is saved. From these preferences the variance computed and saved. The labeled pairs with the highest variance are returned and we exit the labeling process and continue in `main.py`.

### Reward Model Training

The reward model is a simple feedforward neural network with 4 fully connected layers. The input dimension is the sum of the observation and action space dimensions `(obs_dim + action_dim)`. Each of the first three layers has 64 hidden units and uses the ReLU activation function. Additionally, dropout with a rate of 30% is applied after each of these layers to prevent overfitting. The final layer outputs a single scalar value, representing the predicted reward, which is passed through a Tanh activation function to constrain the output to the range [-1, 1] which corresponds to a normalized reward.

The `train_reward_model_ensemble` function (called in `main.py`) is responsible for training an ensemble of reward models using labeled data. Each model is trained independently with its corresponding optimizer. The function supports mini-batch training, validation, and optional logging with TensorBoard. First, the function shuffles the labeled_data at the beginning of each epoch to ensure stochasticity in training. Then, the data is divided into mini-batches of size batch_size. Each reward model in the ensemble is trained independently. To further increase training diversity, the order of the mini-batches is shuffled for each model. During the training process, for each mini-batch, the labeled observation-action pairs are passed through the model in a forward pass, where both segments in each pair are evaluated to predict their respective rewards. The cross-entropy loss is then computed by comparing the predicted probabilities with the true labels. This loss is backpropagated, and the optimizer updates the model’s parameters to minimize the loss.
If validation data (`val_data`) is provided, the function computes the validation loss for each model at the end of every epoch using the compute_reward_model_loss function. We do this to detect potential overfitting. The training and validation losses can be logged using a TensorBoard writer, enabling the visualization of the training process and facilitating debugging or optimization. Once the training process is complete, all updated reward models in the ensemble are used later.

## Client Server Structure
## Results
## Additional Infos