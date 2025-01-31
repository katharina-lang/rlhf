import numpy as np
from sklearn.neighbors import NearestNeighbors

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

        # Fit kNN model on the stored states
        knn = NearestNeighbors(n_neighbors=self.k)
        knn.fit(self.state_buffer)

        # Compute distances for each state in `states`
        distances, _ = knn.kneighbors(states)

        # Estimate density as inverse of the mean kNN distance
        densities = 1 / (np.mean(distances, axis=1) + 1e-6)  # Avoid division by zero
        return densities

def compute_intrinsic_reward(state, density_model):
    """Compute intrinsic reward based on kNN density estimation."""
    state = state.reshape(1, -1)  # Ensure state has the correct shape
    density = density_model.compute_density(state)[0]

    # Calculate intrinsic reward as negative log-density
    intrinsic_reward = -np.log(density + 1e-6)  # Avoid log(0)
    return intrinsic_reward