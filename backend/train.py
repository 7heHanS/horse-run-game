import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os

BOARD_SIZE = 11
NUM_CLASSES = BOARD_SIZE * BOARD_SIZE  # We predict a flattened board index from 0 to 120

class HorseGameDataset(Dataset):
    def __init__(self, csv_file):
        """
        Loads the dataset from the CSV file.
        Input features: The 11x11 board flattened (121 features) + current_player (1 feature) = 122 features.
        Target: The chosen target action coordinate (flattened to a single integer class index: to_r * BOARD_SIZE + to_c)
        """
        self.data = pd.read_csv(csv_file)
        
        # Extract features (cell_0_0 to cell_10_10 + player_turn)
        feature_cols = [f"cell_{y}_{x}" for y in range(BOARD_SIZE) for x in range(BOARD_SIZE)] + ["player_turn"]
        self.X = self.data[feature_cols].values.astype(np.float32)
        
        # Labels: The move destination (acting as a classification task for the best target tile)
        # Note: In a more complex model, we would predict from_tile and to_tile, 
        # but predicting destination is a good start for a simple MLP policy network.
        to_c = self.data["action_to_c"].values
        to_r = self.data["action_to_r"].values
        self.y = (to_r * BOARD_SIZE + to_c).astype(np.int64)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

class AIPolicyNet(nn.Module):
    def __init__(self, input_size=122, hidden_size=256, output_size=121):
        super(AIPolicyNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        out = self.fc3(out)
        return out

def train_model(csv_file="dataset.csv", epochs=15, batch_size=32, model_save_path="model.pt"):
    if not os.path.exists(csv_file):
        print(f"Dataset {csv_file} not found. Please run simulate.py first.")
        return

    print("Loading dataset...")
    dataset = HorseGameDataset(csv_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = AIPolicyNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.2f}%")

    print("Training complete.")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    train_model()
