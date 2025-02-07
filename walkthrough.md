# RLHF Project Documentation

## Table of Contents

- [Setup](#setup)
- [RLHF Functionality](#rlhf-functionality)
  - [Overview](#overview)
  - [Data Collection](#data-collection)
  - [Data Labeling](#data-labeling)
  - [Reward Model Training](#reward-model-training)
  - [Pretraining](#pretraining)
- [Client-Server Architecture](#client-server-architecture)
  - [Overview](#overview-1)
- [Results](#results)
- [Additional Infos](#additional-infos)
  - [Data Collection Examples](#data-collection-examples)

---

## Setup

TBD

---

## RLHF Functionality

### Overview

The structure of our project can be seen in the file [main.py](./rlhf/scripts/main.py). The agent training originates from the CleanRL repository and their file `ppo_continous.py`. While the core structure remains, we modularized the code. In our project, we replaced the environment reward, which is used for the agent-and-critic's training, with a reward predictor.

Key changes:
- The number of policy updates (`num_iterations`) is also the number of reward model updates.
- Data is collected in every iteration.
- Through agent-environment interactions, we gather observations, actions, environment rewards, and predicted rewards.
- The Singleton PPO handles interactions, instantiated in [main.py](./rlhf/scripts/main.py) and defined in [ppo.py](./rlhf.core.ppo.py).
- Data is labeled by a human or a synthetic feedback supplier to provide feedback on actions taken.
- As long as queries are required, `x` pairs are labeled (`x = max(3, 1 + total_queries // num_iterations)`).
- After labeling, the reward model is trained, and the agent and critic are updated.

---

### Data Collection

Before training begins, we create three arrays:
1. Observation-action pairs
2. Environment rewards
3. Predicted rewards

Each training step appends a new index to all three arrays, ensuring corresponding steps align. The function `collect_rollout_data` (from [ppo.py](./rlhf.core.ppo.py)) is called in [main.py](./rlhf/scripts/main.py). After environment interaction and reward prediction, `save_data` stores the collected data.

#### Example Process

Consider two environments, where:
- The observation space dimension is 2.
- The action space dimension is 1.
- Two observation-action pairs are collected.

**Initial state:**
![obs-action-pair](/readme_images/obs_action/pairs_start.png)

**Stacked data:**
![obs-action-buffer](/readme_images/obs_action/pairs_stack.png)

**Final reshaped output:**
![obs-action-output](/readme_images/obs_action/pairs_output.png)

This process applies similarly to predicted and environment rewards. Further examples can be found in [Data Collection Examples](#data-collection-examples).

---

### Data Labeling

After collecting data, labeling is required. Each iteration creates a [labeling instance](./rlhf.core.labeling.py) and calls `get_labeled_data`, returning data in a format similar to the paper *Deep Reinforcement Learning from Human Preferences* ([Christiano et al.](https://arxiv.org/pdf/1706.03741)).

#### Labeling Process
1. **Segment selection:**
   - Random segments are chosen.
   - A segment consists of a trajectory and the sum of all environment rewards (`env_reward`).
2. **Uncertainty-based filtering:**
   - A reward model ensemble (see [Reward Model Training](#reward-model-training)) predicts segment rewards.
   - Variance of predicted rewards determines which pairs are selected.
   - Alternatively, pairs can be selected randomly by toggling a flag.
3. **Labeling Execution:**
   - The function `preference_elicitation` provides selected segments to a human or synthetic labeler.
   - Labeling results in an array of triplets: `(segment1, segment2, (label1, label2))`.

---

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

### Pretraining

TBD

---

## Client-Server Architecture

### Overview

The UI follows a **Client-Server Architecture** using Flask.

- **Backend:** `app.py`, which communicates with `labeling.py`.
- **Frontend:** `index.html`, embedded directly in the Flask server.
- **Workflow:**
  1. `app.py` starts as a thread on a free port when human feedback is needed.
  2. The user clicks a terminal-provided link to open the labeling page.
  3. Two videos appear, and the user selects which one aligns more with the desired behavior.

#### UI Design
![User interface](/readme_images/UI.png)

For further details, see: [Client-Server Architecture](./rlhf/utils/README.md)

---

## Results

### TensorBoard Statistics

Comparison of **HalfCheetah with human feedback (pink) vs. synthetic feedback (yellow):**

![Results](/readme_images/Result.png)

---

## Additional Infos

### Data Collection Examples

#### **Environment Reward**
1. **Collected data (before reshaping):**
   ![env-reward-start](/readme_images/env_reward/env_start.png)
2. **Stacked data:**
   ![env-reward-stack](/readme_images/env_reward/env_stack.png)
3. **Final reshaped output:**
   ![env-reward-output](/readme_images/env_reward/env_output.png)

#### **Predicted Reward**
1. **Collected data:**
   ![pred-reward-start](/readme_images/pred_reward/pred_start.png)
2. **Concatenated representation:**
   ![pred-reward-cat](/readme_images/pred_reward/pred_cat.png)
3. **Final reshaped output:**
   ![pred-reward-output](/readme_images/pred_reward/pred_output.png)

---

This documentation provides an overview of the RLHF project, its modularized approach, data handling, labeling processes, reward model training, and client-server architecture. Additional images and links have been correctly embedded for clarity.

