"""
Training pipeline V3 with Legal Move Masking.
Masks illegal actions before loss computation so the model
learns only strategic choices among valid moves.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
import pandas as pd
import numpy as np
import os

from model_v3 import (
    HorseRunPolicyNetV3, board_to_channels, get_legal_move_mask,
    BOARD_SIZE, NUM_POSITIONS, NUM_ACTIONS
)

OASIS = (5, 5)
MEADOW_SET = {
    (5, 4), (5, 6), (4, 5), (6, 5),
    (5, 3), (5, 7), (3, 5), (7, 5),
    (4, 4), (6, 4), (4, 6), (6, 6)
}


class HorseGameDatasetV3(Dataset):
    """
    Dataset for V3 model with legal move mask pre-computation.
    """
    def __init__(self, csv_file, winner_only=True):
        self.data = pd.read_csv(csv_file)

        if winner_only:
            self.data = self.data[self.data['player_turn'] == self.data['winner']].reset_index(drop=True)

        board_cols = [f"cell_{y}_{x}" for y in range(BOARD_SIZE) for x in range(BOARD_SIZE)]
        self.boards = self.data[board_cols].values.astype(np.float32)
        self.players = self.data["player_turn"].values.astype(np.int64)

        from_c = self.data["action_from_c"].values
        from_r = self.data["action_from_r"].values
        to_c = self.data["action_to_c"].values
        to_r = self.data["action_to_r"].values

        from_idx = from_r * BOARD_SIZE + from_c
        to_idx = to_r * BOARD_SIZE + to_c
        self.labels = (from_idx * NUM_POSITIONS + to_idx).astype(np.int64)

        self.is_critical = self.data["is_critical"].values.astype(np.float32)

        # Pre-compute legal move masks for each sample
        print(f"Pre-computing legal move masks for {len(self.data)} samples...", flush=True)
        self.masks = []
        for i in range(len(self.data)):
            flat = self.boards[i]
            board_2d = flat.reshape(BOARD_SIZE, BOARD_SIZE).tolist()
            # Convert float board to int
            board_int = [[int(c) for c in row] for row in board_2d]
            mask = get_legal_move_mask(board_int, int(self.players[i]))
            self.masks.append(mask)
            if (i + 1) % 5000 == 0:
                print(f"  Masks computed: {i+1}/{len(self.data)}", flush=True)

        print(f"Done. Loaded {len(self.data)} records (winner_only={winner_only})")
        print(f"  Critical moves: {int(self.is_critical.sum())} ({self.is_critical.mean()*100:.1f}%)")
        print(f"  Unique actions: {len(np.unique(self.labels))}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        flat = self.boards[idx]
        board_2d = flat.reshape(BOARD_SIZE, BOARD_SIZE)
        player = self.players[idx]

        opponent = 2 if player == 1 else 1
        channels = np.zeros((3, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        channels[0] = (board_2d == player).astype(np.float32)
        channels[1] = (board_2d == opponent).astype(np.float32)

        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                if (x, y) == OASIS:
                    channels[2, y, x] = 1.0
                elif (x, y) in MEADOW_SET:
                    channels[2, y, x] = 0.5

        return (
            torch.tensor(channels),
            torch.tensor(self.labels[idx]),
            self.masks[idx],  # Pre-computed legal move mask
            torch.tensor(self.is_critical[idx])
        )

    def get_sample_weights(self, critical_weight=10.0):
        weights = np.ones(len(self.data), dtype=np.float32)
        weights[self.is_critical == 1] = critical_weight
        return weights


def masked_cross_entropy_loss(logits, targets, masks):
    """
    Apply legal move masking before computing cross-entropy loss.
    Sets logits of illegal moves to -inf so they get 0 probability after softmax.
    """
    # masks: (batch, 14641) boolean, True = legal
    masked_logits = logits.clone()
    masked_logits[~masks] = -1e9  # illegal moves → -inf probability
    return nn.functional.cross_entropy(masked_logits, targets)


def train_v3(
    csv_file="dataset_v3.csv",
    epochs=50,
    batch_size=256,
    model_save_path="model_v3.pt",
    critical_weight=10.0
):
    if not os.path.exists(csv_file):
        print(f"Dataset {csv_file} not found.")
        return

    print(f"Loading dataset {csv_file}...")
    full_dataset = HorseGameDatasetV3(csv_file, winner_only=True)

    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    sample_weights = full_dataset.get_sample_weights(critical_weight)
    train_indices = train_dataset.indices
    train_weights = sample_weights[train_indices]
    sampler = WeightedRandomSampler(
        weights=train_weights,
        num_samples=len(train_indices),
        replacement=True
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = HorseRunPolicyNetV3(num_res_blocks=4, num_filters=128).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 15

    print(f"\nStarting training V3 + Legal Move Masking ({epochs} epochs)...\n", flush=True)

    for epoch in range(epochs):
        # --- Training ---
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for inputs, labels, masks, _ in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            logits = model(inputs)

            # Apply legal move masking before loss
            loss = masked_cross_entropy_loss(logits, labels, masks)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            # For accuracy, also mask before argmax
            masked_logits = logits.clone()
            masked_logits[~masks] = -1e9
            _, predicted = torch.max(masked_logits, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        # --- Validation ---
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels, masks, _ in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                masks = masks.to(device)

                logits = model(inputs)
                loss = masked_cross_entropy_loss(logits, labels, masks)
                val_loss += loss.item()

                masked_logits = logits.clone()
                masked_logits[~masks] = -1e9
                _, predicted = torch.max(masked_logits, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_acc = 100 * train_correct / train_total if train_total > 0 else 0
        val_acc = 100 * val_correct / val_total if val_total > 0 else 0
        lr = optimizer.param_groups[0]['lr']

        print(
            f"Epoch {epoch+1:3d}/{epochs} | "
            f"Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.1f}% | "
            f"Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.1f}% | "
            f"LR: {lr:.6f}",
            flush=True
        )

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"  --> Best model saved to {model_save_path}", flush=True)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"Early stopping at epoch {epoch+1}", flush=True)
                break

    print("\nTraining V3 + Masking Complete.", flush=True)


if __name__ == "__main__":
    train_v3()
