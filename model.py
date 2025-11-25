import torch.nn as nn

def build_model(input_dim):
    model = nn.Sequential(
        nn.Linear(input_dim, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 2)   # Binary classification
    )
    return model