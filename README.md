# Rlhf Project Documentation

- [Setup](#setup)
- [RLHF functionality](#rlhf-functionality)
- [Overview](#overview)
- [Data Collection](#data-collection)
- [Data Labeling](#data-labeling)
- [Reward Model Training](#reward-model-training)
- [Pretraining](#pretraining)
- [Client-Server Architecture](#client-server-architecture)
- [Results](#results)
- [Additional Infos](#additional-infos)


## Setup

## RLHF Functionality

### Overview

The structure of our Project can be seen in the file `rlhf/scripts/main.py`. The agent training has been taken from the CleanRL repository and their file `ppo_continous.py`. The core structure of their file remains but it is modularized. In our project we replaced the environment reward which is used for the agents (and critics) training with a reward predictor.
The number of policy updates (num_iterations) also is the number of reward model updates. Every iteration, data is collected. The agent interacts with the environment and through this we get observations with their corresponding actions, the environment rewards for an action and our predicted reward. Interaction with the environment is handled through the Singleton PPO which is instantiated in `main.py` and defined in `rlhf.core.ppo.py`.
Then we label the data. As long as we still want queries, x pairs get labeled (x = max(3, 1+total_queries//num_iterations)). After the labeling process the reward model is trained and then agent and critic are updated.

### Data Collection

Per data category (observation action pair, environment_reward, predicted_reward) we want an array. The indices between these should match. From the file `ppo.py` the function `collect_rollout_data` gets called in `main.py`.
After the environment interaction and reward prediction `save_data` saves the data into arrays.
The data is stacked first and and reshaped at the end. To make it easier to understand how the data is collected an example is provided.
In the following example we have two environments and visualize the process for the observation action buffer. The observation space dimension is two and the action space dimension is one.
Here two observation action pairs are collected (because of the two environments), and a pair consists of the concatenated observation and its action e.g. [1 2 3].
The collected data looks like the following.

![obs-action-pair](/readme_images/obs_action/pairs_start.png)

When new data is collected, it is stacked with np.hstack and one row corresponds to one environment:

![obs-action-buffer](/readme_images/obs_action/pairs_stack.png)

After the agent environment interaction is finished for the whole iteration the data arrays get reshaped. The environments data basically gets concatenated and the original observation action pairs get reconstructed.

![obs-action-output](/readme_images/obs_action/pairs_output.png)

Like this we ensure that each index corresponds to the related data.
The same process applies to the predicted and environment rewards, with slight modifications. This ensures that each index i now aligns across all arrays. The corresponding examples for the environment and predicted reward are provided in [Data collection examples](#data-collection-examples).



### Data Labeling
Every iteration a Labeling instance is created and it's function `get_labeled_data` is called. This returns the labeled data in a format similar to the one in the paper "Deep reinforcement learning from human preferences".
An array of triplets is returned here. One triplet looks like the following: (trajectory1, trajectory2, (label1, label2)). A trajectory consists of n observation action pairs (n=segment_size).
Through `get_labeled_data()` random segments get selected and a segment is a tuple of the trajectory and the env_reward. The environment reward is the sum over all env_rewards for the corresponding observation action pairs.
After segment selection, the wanted number of labeled pairs is created. Per default, an uncertainty-based method is used but it is also possible to randomly select these pairs.
For the uncertainty-based approach, every reward model (there is an ensemble if wanted, more about this in [Reward Model Training](#reward-model-training)) predicts the reward for the trajectories. The sum is taken and the preference per model is saved. From these preferences, the variance is computed and saved. The labeled pairs with the highest variance are returned and we exit the labeling process and continue in `main.py`.

### Reward Model Training

The reward model is a simple feedforward neural network with 4 fully connected layers. The input dimension is the sum of the observation and action space dimensions `(obs_dim + action_dim)`. Each of the first three layers has 64 hidden units and uses the ReLU activation function. Additionally, dropout with a rate of 30% is applied after each of these layers to prevent overfitting. The final layer outputs a single scalar value, representing the predicted reward, which is passed through a Tanh activation function to constrain the output to the range [-1, 1] which corresponds to a normalized reward.

The `train_reward_model_ensemble` function (called in `main.py`) is responsible for training an ensemble of reward models using labeled data. Each model is trained independently with its corresponding optimizer. The function supports mini-batch training, validation, and logging with TensorBoard. First, the function shuffles the labeled_data at the beginning of each epoch to ensure stochasticity in training. Then, the data is divided into mini-batches of size batch_size. Each reward model in the ensemble is trained independently. To further increase training diversity, the order of the mini-batches is shuffled for each model. During the training process, for each mini-batch, the labeled observation-action pairs are passed through the model in a forward pass, where both segments in each pair are evaluated to predict their respective rewards. The cross-entropy loss is then computed by comparing the predicted probabilities with the true labels. This loss is backpropagated, and the optimizer updates the model’s parameters to minimize the loss.
If validation data (`val_data`) is provided, the function computes the validation loss for each model at the end of every epoch using the compute_reward_model_loss function. We do this to detect potential overfitting. The training and validation losses are logged using a TensorBoard writer, enabling the visualization of the training process and facilitating debugging or optimization. Once the training process is complete, all updated reward models in the ensemble are used later.

### Pretraining


## Client-Server Architecture

### Overview

We developed our UI as a Client-Server Architecture with a Flask backend and embedded the frontend directly in it. <br>
Our UI includes the frontend index.html and the backend app.py, which communicates with labeling.py. In main, app.py is started as a thread on a free port found in the backend as soon as human feedback is expected. <br>
The human labels the two displayed videos using buttons, indicating which or if one looks more like the desired behavior. <br>
The user interface is designed the following way:
![User interface](/readme_images/UI.png)

### Structure

The frontend is responsible for displaying the user interface; the videos are passed from the backend to the frontend and get displayed by it. <br>
When a button for labeling is clicked, the backend is informed about the choice. <br>
The frontend is also responsible for disabling the buttons and displaying a loading message while the next videos are being recorded and the human has nothing to evaluate.

The backend is responsible for the logic behind the user interface. <br>
It processes the button clicks from the frontend and sends the resulting label to labeling.py. <br>
After that, it fetches the new videos from the designated uploads folder. <br>
It also checks whether labeling is complete and, if so, terminates the Flask thread.

If human feedback is desired, labeling.py creates a folder for uploading videos in preference_elicitation(). It checks to make sure that the folder is empty and, if not, empties it. <br>
After that, the videos for the two current segments are recorded in record_segments.py. To do this, the segments are split into their individual observations and actions, which are then executed step by step and recorded. When this is complete and the two videos are in the uploads folder, app.py is notified. <br>
As soon as the frontend displays the videos, a human can label them. The label is stored in a tuple in the backend and a boolean variable is set when a new label becomes available. As soon as this boolean variable is True, labeling.py fetches the label from app.py and get_labeled_data() appends it to the labeled_data list so that the reward models can be trained with it.

Incomplete sequence diagram of the process:
![Incomplete sequence diagram of the process](/readme_images/SequenzUI.png)

### Results

TensorBoard statistics for HalfCheetah with human feedback (pink) and with synthetic feedback (yellow):
![TensorBoard statistics for HalfCheetah with human feedback (pink) and with synthetic feedback (yellow)](/readme_images/Result.png)


## Results
## Additional Infos
### Data collection examples
For the following ecamples the number of enironments is also two.
#### environment reward
The data gets collected and reshaped immediately so we can use the same procedure as for the observation_action pairs.

![env-reward-start](/readme_images/env_reward/env_start.png)

Then the reshaped data gets stacked on top of each other.

![env-reward-stack](/readme_images/env_reward/env_stack.png)

At last, the reshape process flattens the array so that the rewards are concatenated between environments.

![env-reward-output](/readme_images/env_reward/env_output.png)

#### predicted reward
The difference here is, that the data is saved as a tensor instead of an numpy array. The rest is analog again.

![pred-reward-start](/readme_images/pred_reward/pred_start.png)

![pred-reward-cat](/readme_images/pred_reward/pred_cat.png)

![pred-reward-output](/readme_images/pred_reward/pred_output.png)

