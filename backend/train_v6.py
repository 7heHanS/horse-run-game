"""
V6 Fine-Tuning Pipeline.
Loads V5 model and fine-tunes on combined dataset:
  - Original MCTS overnight dataset (general strength)
  - Targeted side-attack defense dataset (vulnerability patch)

The targeted data receives higher critical_weight to ensure
the model learns defensive patterns against side-column attacks.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split, WeightedRandomSampler
import pandas as pd
import numpy as np
import os

from model_v4 import (
    HorseRunPolicyNetV4, board_to_channels, get_v4_masks,
    BOARD_SIZE, NUM_POSITIONS, OASIS, MEADOW_SET
)

class HorseGameDatasetV6(Dataset):
    def __init__(self, csv_file, winner_only=True, extra_critical_boost=1.0):
        """
        extra_critical_boost: multiplier on top of is_critical for this dataset.
        Used to give targeted data extra weight.
        """
        self.extra_boost = extra_critical_boost
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

        print(f"Loading {csv_file}: {len(self.data)} records (boost={extra_critical_boost}x)", flush=True)
        self.from_masks = []
        self.to_masks = []
        for i in range(len(self.data)):
            flat = self.boards[i]
            board_2d = flat.reshape(BOARD_SIZE, BOARD_SIZE).tolist()
            board_int = [[int(c) for c in row] for row in board_2d]
            
            target_from = int(self.from_labels[i])
            f_mask, t_mask = get_v4_masks(board_int, int(self.players[i]), target_from)
            
            self.from_masks.append(f_mask)
            self.to_masks.append(t_mask)

        print(f"  Done. {len(self.data)} records loaded.", flush=True)

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

    def get_sample_weights(self, critical_weight=50.0):
        weights = np.ones(len(self.data), dtype=np.float32) * self.extra_boost
        critical_mask = self.is_critical == 1
        weights[critical_mask] = critical_weight * self.extra_boost
        return weights


def masked_cross_entropy_loss(logits, targets, masks):
    masked_logits = logits.clone()
    masked_logits[~masks] = -1e9
    return nn.functional.cross_entropy(masked_logits, targets)


def get_accuracy(logits, targets, masks):
    masked_logits = logits.clone()
    masked_logits[~masks] = -1e9
    _, predicted = torch.max(masked_logits, 1)
    correct = (predicted == targets).sum().item()
    return correct, targets.size(0)


def train_v6(
    base_dataset="dataset_mcts_overnight.csv",
    targeted_dataset="dataset_targeted_side.csv",
    base_model="model_v5.pt",
    epochs=10,
    batch_size=128,
    model_save_path="model_v6.pt"
):
    datasets = []

    # Load base overnight dataset (general strength)
    if os.path.exists(base_dataset):
        ds_base = HorseGameDatasetV6(base_dataset, winner_only=True, extra_critical_boost=1.0)
        datasets.append(ds_base)
    else:
        print(f"Warning: {base_dataset} not found. Training on targeted data only.")

    # Load targeted dataset (side-attack defense) with 25x boost
    if os.path.exists(targeted_dataset):
        ds_targeted = HorseGameDatasetV6(targeted_dataset, winner_only=True, extra_critical_boost=25.0)
        datasets.append(ds_targeted)
    else:
        print(f"Error: {targeted_dataset} not found. Run generate_targeted_dataset.py first.")
        return

    if not datasets:
        print("No datasets available. Aborting.")
        return

    # Combine datasets
    full_dataset = ConcatDataset(datasets)
    total_size = len(full_dataset)
    
    train_size = int(0.9 * total_size)
    val_size = total_size - train_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Build weighted sampler using combined weights
    all_weights = []
    for ds in datasets:
        w = ds.get_sample_weights(critical_weight=80.0)
        all_weights.append(w)
    combined_weights = np.concatenate(all_weights)
    
    train_indices = train_dataset.indices
    train_weights = combined_weights[train_indices]
    sampler = WeightedRandomSampler(
        weights=train_weights, 
        num_samples=len(train_indices), 
        replacement=True
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HorseRunPolicyNetV4().to(device)

    # Load V5 weights
    if os.path.exists(base_model):
        print(f"Loading base weights from {base_model}...")
        model.load_state_dict(torch.load(base_model, map_location=device, weights_only=True))
    else:
        print(f"Warning: {base_model} not found. Training from scratch.")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nV6 Fine-tuning: {total_params:,} params")
    print(f"  Base data: {len(ds_base) if 'ds_base' in dir() else 0} records")
    print(f"  Targeted data: {len(ds_targeted)} records (3x boost)")
    print(f"  Total: {total_size} records")
    print(f"  Train: {train_size}, Val: {val_size}")
    print(f"  Device: {device}")

    # Smaller learning rate for targeted fine-tuning
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

    best_val_loss = float('inf')
    patience_counter = 0

    print("\nStarting V6 Fine-Tuning...\n", flush=True)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_f_corr, train_t_corr, train_total = 0, 0, 0

        for inputs, f_labels, t_labels, f_masks, t_masks, _ in train_loader:
            inputs = inputs.to(device)
            f_labels, t_labels = f_labels.to(device), t_labels.to(device)
            f_masks, t_masks = f_masks.to(device), t_masks.to(device)

            optimizer.zero_grad()
            logits_from, logits_to = model(inputs)

            loss_from = masked_cross_entropy_loss(logits_from, f_labels, f_masks)
            loss_to = masked_cross_entropy_loss(logits_to, t_labels, t_masks)
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
                inputs = inputs.to(device)
                f_labels, t_labels = f_labels.to(device), t_labels.to(device)
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
            f"Epoch {epoch+1:2d}/{epochs} | "
            f"Loss {train_loss/len(train_loader):.3f} (F:{100*train_f_corr/train_total:.1f}% T:{100*train_t_corr/train_total:.1f}%) | "
            f"Val Loss {avg_val_loss:.3f} (F:{100*val_f_corr/val_total:.1f}% T:{100*val_t_corr/val_total:.1f}%) | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}",
            flush=True
        )

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"  --> Best Model V6 saved!", flush=True)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 5:
                print(f"Early stopping at epoch {epoch+1}", flush=True)
                break

    print(f"\nTraining complete. Best model saved as {model_save_path}")


if __name__ == "__main__":
    train_v6()
