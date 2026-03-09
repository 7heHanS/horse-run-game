"""
Training pipeline V4 (Multi-Head + Legal Move Masking).
Trains the From Head and To Head independently with segregated masks.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
import pandas as pd
import numpy as np
import os

from model_v4 import (
    HorseRunPolicyNetV4, board_to_channels, get_v4_masks,
    BOARD_SIZE, NUM_POSITIONS, OASIS, MEADOW_SET
)

class HorseGameDatasetV4(Dataset):
    """
    Dataset for V4 model with segregated `from` and `to` masked targets.
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

        self.from_labels = (from_r * BOARD_SIZE + from_c).astype(np.int64)
        self.to_labels = (to_r * BOARD_SIZE + to_c).astype(np.int64)
        self.is_critical = self.data["is_critical"].values.astype(np.float32)

        print(f"Pre-computing segregated V4 masks for {len(self.data)} samples...", flush=True)
        self.from_masks = []
        self.to_masks = []
        for i in range(len(self.data)):
            flat = self.boards[i]
            board_2d = flat.reshape(BOARD_SIZE, BOARD_SIZE).tolist()
            board_int = [[int(c) for c in row] for row in board_2d]
            
            # Use the target_from_idx to get the SPECIFIC valid destinations for the piece that was ACTUALLY moved
            target_from = int(self.from_labels[i])
            f_mask, t_mask = get_v4_masks(board_int, int(self.players[i]), target_from)
            
            self.from_masks.append(f_mask)
            self.to_masks.append(t_mask)
            
            if (i + 1) % 5000 == 0:
                print(f"  Masks computed: {i+1}/{len(self.data)}", flush=True)

        print(f"Done. Loaded {len(self.data)} records.")

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
            torch.tensor(self.from_labels[idx]),
            torch.tensor(self.to_labels[idx]),
            self.from_masks[idx], 
            self.to_masks[idx],
            torch.tensor(self.is_critical[idx])
        )

    def get_sample_weights(self, critical_weight=10.0):
        weights = np.ones(len(self.data), dtype=np.float32)
        weights[self.is_critical == 1] = critical_weight
        return weights


def masked_cross_entropy_loss(logits, targets, masks):
    masked_logits = logits.clone()
    masked_logits[~masks] = -1e9  # illegal moves -> -inf
    return nn.functional.cross_entropy(masked_logits, targets)


def get_accuracy(logits, targets, masks):
    masked_logits = logits.clone()
    masked_logits[~masks] = -1e9
    _, predicted = torch.max(masked_logits, 1)
    correct = (predicted == targets).sum().item()
    return correct, targets.size(0)


def train_v4(csv_file="dataset_v3.csv", epochs=50, batch_size=256, model_save_path="model_v4.pt"):
    if not os.path.exists(csv_file):
        print(f"Dataset {csv_file} not found.")
        return

    full_dataset = HorseGameDatasetV4(csv_file, winner_only=True)
    
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    sample_weights = full_dataset.get_sample_weights(critical_weight=10.0)
    train_indices = train_dataset.indices
    train_weights = sample_weights[train_indices]
    sampler = WeightedRandomSampler(weights=train_weights, num_samples=len(train_indices), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HorseRunPolicyNetV4().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"V4 Model params: {total_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    best_val_loss = float('inf')
    patience_counter = 0

    print("\nStarting V4 Multi-Head Training...", flush=True)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_f_corr, train_t_corr, train_total = 0, 0, 0

        for inputs, f_labels, t_labels, f_masks, t_masks, _ in train_loader:
            inputs, f_labels, t_labels = inputs.to(device), f_labels.to(device), t_labels.to(device)
            f_masks, t_masks = f_masks.to(device), t_masks.to(device)

            optimizer.zero_grad()
            logits_from, logits_to = model(inputs)

            loss_from = masked_cross_entropy_loss(logits_from, f_labels, f_masks)
            loss_to = masked_cross_entropy_loss(logits_to, t_labels, t_masks)
            
            # Loss balancing: predicting "From" is easier (1 of ~10) than "To" (1 of ~121)
            loss = 0.5 * loss_from + 1.0 * loss_to
            
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            c_f, tot = get_accuracy(logits_from, f_labels, f_masks)
            c_t, _ = get_accuracy(logits_to, t_labels, t_masks)
            train_f_corr += c_f
            train_t_corr += c_t
            train_total += tot

        # Validation
        model.eval()
        val_loss = 0
        val_f_corr, val_t_corr, val_total = 0, 0, 0

        with torch.no_grad():
            for inputs, f_labels, t_labels, f_masks, t_masks, _ in val_loader:
                inputs, f_labels, t_labels = inputs.to(device), f_labels.to(device), t_labels.to(device)
                f_masks, t_masks = f_masks.to(device), t_masks.to(device)

                logits_from, logits_to = model(inputs)
                l_f = masked_cross_entropy_loss(logits_from, f_labels, f_masks)
                l_t = masked_cross_entropy_loss(logits_to, t_labels, t_masks)
                loss = 0.5 * l_f + 1.0 * l_t
                val_loss += loss.item()

                c_f, tot = get_accuracy(logits_from, f_labels, f_masks)
                c_t, _ = get_accuracy(logits_to, t_labels, t_masks)
                val_f_corr += c_f
                val_t_corr += c_t
                val_total += tot

        avg_val_loss = val_loss / len(val_loader)
        
        print(
            f"Epoch {epoch+1:2d} | "
            f"Train Loss {train_loss/len(train_loader):.3f} (F:{100*train_f_corr/train_total:.1f}% T:{100*train_t_corr/train_total:.1f}%) | "
            f"Val Loss {avg_val_loss:.3f} (F:{100*val_f_corr/val_total:.1f}% T:{100*val_t_corr/val_total:.1f}%) | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}",
            flush=True
        )

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"  --> Best model saved", flush=True)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 15:
                print(f"Early stopping at epoch {epoch+1}", flush=True)
                break

if __name__ == "__main__":
    train_v4()
