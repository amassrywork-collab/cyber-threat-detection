import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from model import build_model

# ============================
# Load dataset
# ============================
train_df = pd.read_csv("data/labelled_train.csv")
val_df   = pd.read_csv("data/labelled_validation.csv")

X_train = train_df.drop(columns=['sus_label']).values
y_train = train_df['sus_label'].values

X_val = val_df.drop(columns=['sus_label']).values
y_val = val_df['sus_label'].values

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)

X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.long)

train_ds = TensorDataset(X_train, y_train)
val_ds   = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=32)


# ============================
# Build model
# ============================
input_dim = X_train.shape[1]
model = build_model(input_dim)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=2)

# ============================
# Training Loop (10 epochs)
# ============================
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    for X, y in train_loader:
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    accuracy_metric.reset()

    with torch.no_grad():
        for X, y in val_loader:
            pred = model(X)
            preds = pred.argmax(dim=-1)
            accuracy_metric.update(preds, y)

    val_accuracy = accuracy_metric.compute().item()
    print(f"Epoch {epoch+1}/{num_epochs} — Validation Accuracy: {val_accuracy:.4f}")

# Required output
val_accuracy = float(val_accuracy)

print("Final Validation Accuracy:", val_accuracy)
