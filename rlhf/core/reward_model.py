import numpy as np
import torch
import torch.nn as nn


# input dim is concatenated state, action
# how Do i want to concatenate? a segment are multiple state action pairs
# each row is a concatenated obs, action
# observation, action, observation, action ?
# for now, segments are 60 obs, action pairs
# matrix with 60 rows

# Instanz des Mosells    

class RewardModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(RewardModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.model(x)



# Loss Function

    def preference_loss(reward1, reward2, preference):
        """
        Berechnet den Cross-Entropy-Loss für Präferenzen.
        """
        prob1 = torch.exp(reward1) / (torch.exp(reward1) + torch.exp(reward2))
        prob2 = 1 - prob1
    
        # Log-Likelihood: Cross Entropy Loss
        loss = -preference * torch.log(prob1 + 1e-8) - (1 - preference) * torch.log(prob2 + 1e-8)
        return loss.mean()



    # Training: Der Optimierungsprozess mit Regularisierung und Validierung 

    # Optimizer und Regularisierung
    optimizer = optim.Adam(reward_model.parameters(), lr=0.001)
    l2_regularization = 1e-4


    # Training

    num_epochs = 10
    for epoch in range(num_epochs):
        epoch_loss = 0
        for traj1, traj2, pref in dataloader:
            traj1 = torch.tensor(traj1, dtype=torch.float32)
            traj2 = torch.tensor(traj2, dtype=torch.float32)
            pref = torch.tensor(pref, dtype=torch.float32)
        
            # Vorhersage der Belohnungen
            reward1 = reward_model(traj1).squeeze()
            reward2 = reward_model(traj2).squeeze()
        
            # Verlustberechnung
            loss = preference_loss(reward1, reward2, pref)
        
            # L2-Regularisierung hinzufügen
            l2_loss = sum(p.pow(2.0).sum() for p in reward_model.parameters())
            loss += l2_regularization * l2_loss
        
            # Backpropagation und Parameter-Update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            epoch_loss += loss.item()
    
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}")


    # Validierung (Overfitting erkennen)

    def validate_model(model, dataloader):
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for traj1, traj2, pref in dataloader:
                traj1 = torch.tensor(traj1, dtype=torch.float32)
                traj2 = torch.tensor(traj2, dtype=torch.float32)
                pref = torch.tensor(pref, dtype=torch.float32)
            
                # Vorhersagen
                reward1 = model(traj1).squeeze()
                reward2 = model(traj2).squeeze()
            
                # Verlustberechnung
                loss = preference_loss(reward1, reward2, pref)
                total_loss += loss.item()
    
        print(f"Validation Loss: {total_loss:.4f}")
        model.train()

    # Validierung durchführen
    validate_model(reward_model, dataloader)







# Instanz des Mosells    
reward_model = RewardModel(input_size=10)