import torch
import torch.nn as nn
import torch.optim as optim

# Einfacher Autoencoder
class StateDensityModel(nn.Module):
    def __init__(self, input_dim):
        super(StateDensityModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 16)
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

# Autoencoder initialisieren
def initialize_autoencoder(envs):
    state_dim = envs.single_observation_space.shape
    input_dim = state_dim[0] if len(state_dim) == 1 else int(np.prod(state_dim))
    density_model = StateDensityModel(input_dim=input_dim)
    optimizer = optim.Adam(density_model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    return density_model, optimizer, loss_fn

# Funktion für intrinsische Belohnung basierend auf Rekonstruktionsfehler
def compute_intrinsic_reward(state,density_model,loss_fn):
    state_tensor = torch.tensor(state, dtype=torch.float32)
    reconstructed = density_model(state_tensor)
    # Compute the difference between reconstructed and real state
    # --> higher reconstruction_error for less known states and thus higher intrinsic reward
    reconstruction_error = loss_fn(reconstructed, state_tensor).item()
    print("Reconstruction Error: " + str(reconstruction_error))
    # Zustandsdichte p(s) schätzen (1/Rekonstruktionsfehler kann als Proxy dienen)
    # Verhindere Division durch 0: minimaler Fehler (epsilon)
    epsilon = 1e-6
    state_density = 1 / (reconstruction_error + epsilon)
    print("State Density: " + str(state_density))
    
    # Berechne den intrinsischen Reward: r_intr(s) = -log(p(s))
    intrinsic_reward = -torch.log(torch.tensor(state_density)).item()
    print("Intrinsic Reward: " + str(intrinsic_reward))
    return intrinsic_reward