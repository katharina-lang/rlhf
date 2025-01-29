import numpy as np

class KNNDensityModel:
    def __init__(self, k=5):
        self.k = k
        self.state_buffer = []  # Buffer to store observed states

    def add_states(self, states):
        """Add states to the buffer for kNN computation."""
        self.state_buffer.extend(states)

    def compute_density(self, states):
        """Estimate state densities using kNN distances."""
        if len(self.state_buffer) < self.k:
            return np.zeros(len(states))  # Avoid division by zero

        # Convert buffer to NumPy array for distance computation
        buffer_array = np.array(self.state_buffer)
        densities = []

        for state in states:
            # Compute Euclidean distances to all states in the buffer
            distances = np.linalg.norm(buffer_array - state, axis=1)
            # Sort distances and take the k-th smallest distance
            k_nearest_distances = np.sort(distances)[:self.k]
            # Estimate density as inverse of the mean kNN distance
            density = 1 / (np.mean(k_nearest_distances) + 1e-6)  # Avoid division by zero
            densities.append(density)

        return np.array(densities)

def compute_intrinsic_reward(state, density_model):
    """Compute intrinsic reward based on kNN density estimation."""
    state = state.reshape(1, -1)  # Ensure state has the correct shape
    density = density_model.compute_density(state)[0]

    # Calculate intrinsic reward as negative log-density
    intrinsic_reward = -np.log(density + 1e-6)  # Avoid log(0)
    print(f"Intrinsic reward: {intrinsic_reward}")
    return intrinsic_reward
