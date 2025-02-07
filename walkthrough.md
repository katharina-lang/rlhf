# RLHF Project Documentation

## Table of Contents

- [RLHF Functionality](#rlhf-functionality)
  - [Overview](#overview)
  - [Data Collection](#data-collection)
  - [Data Labeling](#data-labeling)
  - [Reward Model Training](#reward-model-training)
  - [Unsupervised pretraining](#Unsupervised-pretraining)
- [Client-Server Architecture](#client-server-architecture)
  - [Overview](#overview-1)
- [Additional Infos](#additional-infos)
  - [Data Collection Examples](#data-collection-examples)

---

## RLHF Functionality

### Overview

The structure of our project can be seen in the file [main.py](./rlhf/main.py). The agent training has been taken from the CleanRL repository and their file `ppo_continous.py`. The core structure of their file has remained, but it is now modularized. In our project, we replaced the environment reward - which is used for the agent-and-critic's training - with a reward predictor.
The number of policy updates (num_iterations) is also the number of reward model updates. Data is collected in every iteration. Through the agent's interactions with the environment we receive observations as well as their corresponding actions, the environment rewards for each action and our predicted reward. Interaction with the environment is handled through the Singleton PPO which is instantiated in [main.py](./rlhf/main.py) and defined in [ppo.py](./rlhf/core/ppo.py).
The data is then labeled by either a human or a synthetic feedback supplier in order to receive feedback on the actions that were taken. As long as queries are still required, x pairs are labeled (x = max(3, 1+total_queries//num_iterations)). After the labeling process, the reward model is trained and then agent as well as critic are updated.

---

### Data Collection

Before training begins, we create three arrays:
1. Observation-action pairs
2. Environment rewards
3. Predicted rewards

Throughout the data collection, each training step adds a new entry to each of these arrays so the indices for corresponding steps match between all three arrays. 
In order to start our data collection, we call the function `collect_rollout_data` from [ppo.py](./rlhf/core/ppo.py) in [main](./rlhf/main.py).
After the environment interaction and reward prediction, `save_data` saves the collected data into the above-mentioned arrays.

We stack the data first and then reshape it at the end. The following example illustrates the data collection process:

Consider two environments, where:
- The observation space dimension is 2.
- The action space dimension is 1.
- Two observation-action pairs are collected.

**Initial state:**

![obs-action-pair](/documents/obs_action/pairs_start.png)

For each step, the collected data is stacked with np.hstack and each row corresponds to all observation-action pairs in their correct order collected by one environment over the course of `collect_rollout_data`:

**Stacked data:**

![obs-action-buffer](/documents/obs_action/pairs_stack.png)

After the agent-environment interaction is finished for the entire iteration, the data arrays are reshaped in such a way that the original observation-action pairs are reconstructed.

**Final reshaped output:**

![obs-action-output](/documents/obs_action/pairs_output.png)

The same process applies to the predicted and environment rewards, with slight modifications. This ensures that each index now aligns across all arrays. The corresponding examples for the environment and predicted reward are provided in the [Data collection examples](#data-collection-examples) section.

---

### Data Labeling

After collecting data, labeling is required. Each iteration creates a [labeling instance](./rlhf/core/labeling.py) and calls `get_labeled_data`, returning data in a format similar to the paper *Deep Reinforcement Learning from Human Preferences* ([Christiano et al.](https://arxiv.org/pdf/1706.03741)).

By calling `get_labeled_data()`, random segments are selected. A segment is a tuple of the trajectory and the the sum over all environment rewards for the corresponding observation-action pairs (env_reward).
We extract the amount of segments from the randomly selected segments. Per default, an uncertainty-based method is used, but it is also possible to randomly select these pairs by setting the corresponding flag.
For the uncertainty-based approach, a reward model ensemble (more about this in [Reward Model Training](#reward-model-training)) predicts the rewards for the segments. The sum of predicted rewards for each segment is taken and the preference per model is saved. From these preferences, the variance is computed. The pairs with the highest variance are returned. 
The actual labeling process is set to work by calling the function `preference_elicitation` which provides the selected segments to either a human or a synthetic labeler for feedback.
We exit the labeling process and continue in [main.py](./rlhf/main.py).
An array of triplets which each looks like the following (segment1, segment2, (label1, label2)) is returned.


### Reward Model Training

The reward model is a feedforward neural network with:
- **4 fully connected layers**
- **ReLU activation** in the first 3 layers (each with 64 hidden units)
- **Dropout (30%)** to prevent overfitting
- **Tanh activation** in the final layer (normalizing output to [-1,1])

#### Training Process
- Called via `train_reward_model_ensemble` in `main.py`.
- Training supports **mini-batches, validation, and logging via TensorBoard**.
- Data is **shuffled** at the start of each epoch for stochasticity.
- **Mini-batches** of size `batch_size_rm` are created.
- Each model in the ensemble trains **independently** with its optimizer.
- **Batch order is shuffled** per model for additional diversity.
- **Cross-entropy loss** is used, updating parameters via backpropagation.

---

### Unsupervised Pretraining

The overall goal of unsupervised pretraining is to train the policy to explore the environment as much as possible before the actual training process. This leads to the agent showing more diverse behavior due to increased entropy. We use 1% of the overall iterations for pretraining as indicated in the Pebble paper (Lee et al., 2021). The intrinsic rewards are computed using the k-th nearest neighbor method. This functionality can be found in `unsupervised_pt.py` in the core module. Pretraining can be activated by setting the flag `unsupervised_pretraining`.

---

## Client-Server Architecture

### Overview
We developed our UI as a Client-Server Architecture with a Flask backend and embedded the frontend directly in it.

Our UI includes the frontend index.html and the backend app.py, which communicates with labeling.py. In main, app.py is started as a thread on a free port found in the backend as soon as human feedback is expected. Then the user can click on the link to the port in the terminal to open the labeling page. <br>
The human labels the two displayed videos using buttons, indicating which or if one looks more like the desired behavior. <br>
The user interface is designed the following way:
![User interface](/documents/UI.png)

For more detailed information, see: 
- [Client-Server Architecture](./rlhf/utils/README.md)

---

## Additional Infos

### Data collection examples
The number of environments for the following examples is two.
#### Environment reward
The data is collected and reshaped immediately, so we can use the same procedure as for the observation-action pairs.

![env-reward-start](/documents/env_reward/env_start.png)

Then the reshaped data is stacked.

![env-reward-stack](/documents/env_reward/env_stack.png)

At last, the reshape process flattens the array so that the rewards are concatenated between environments.

![env-reward-output](/documents/env_reward/env_output.png)

#### Predicted reward
The difference here is that the data is saved as a tensor instead of a numpy array. The rest is analogous.

![pred-reward-start](/documents/pred_reward/pred_start.png)

![pred-reward-cat](/documents/pred_reward/pred_cat.png)

![pred-reward-output](/documents/pred_reward/pred_output.png)

---

This documentation provides an overview of the RLHF project, its modularized approach, data handling, labeling processes, reward model training, and client-server architecture. Additional images and links have been correctly embedded for clarity.
