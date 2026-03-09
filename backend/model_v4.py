"""
Horse Run Game - AI Policy Network V4 (Multi-Head Architecture)
Splits the 14641 action space into two independent 121-class heads:
- Head A: Which piece to move (from_idx)
- Head B: Where to move it (to_idx)
"""
import torch
import torch.nn as nn

BOARD_SIZE = 11
NUM_POSITIONS = BOARD_SIZE * BOARD_SIZE  # 121

OASIS = (5, 5)
MEADOW_SET = {
    (5, 4), (5, 6), (4, 5), (6, 5),
    (5, 3), (5, 7), (3, 5), (7, 5),
    (4, 4), (6, 4), (4, 6), (6, 6)
}

def board_to_channels(board, current_player):
    """
    Converts a dataset row or 2D board + player into a (3, 11, 11) tensor.
    Channel 0: Current player's pieces (1)
    Channel 1: Opponent's pieces (1)
    Channel 2: Special tiles (Oasis=1.0, Meadow=0.5)
    """
    opponent = 2 if current_player == 1 else 1
    channels = torch.zeros(3, BOARD_SIZE, BOARD_SIZE)
    
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            cell = board[y][x] if isinstance(board[y], list) else board[y * BOARD_SIZE + x]
            if cell == current_player:
                channels[0, y, x] = 1.0
            elif cell == opponent:
                channels[1, y, x] = 1.0
            
            if (x, y) == OASIS:
                channels[2, y, x] = 1.0
            elif (x, y) in MEADOW_SET:
                channels[2, y, x] = 0.5
    
    return channels


def get_v4_masks(board_2d, current_player, target_from_idx=None):
    """
    Generates segregated legal move masks for V4.

    For Inference: Call without target_from_idx. Returns just the `from_mask`.
    For Training: Call WITH target_from_idx. Returns `from_mask` and a `to_mask` 
                  that is ONLY valid for the specified target_from_idx.
    
    returns:
      from_mask: (121,) boolean tensor. True = piece can be moved.
      to_mask: (121,) boolean tensor. True = valid destination for target_from_idx.
               (Returns None if target_from_idx is None)
    """
    from engine import get_valid_slide_moves, get_valid_l_shape_moves
    
    from_mask = torch.zeros(NUM_POSITIONS, dtype=torch.bool)
    to_mask = torch.zeros(NUM_POSITIONS, dtype=torch.bool) if target_from_idx is not None else None
    
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            if board_2d[y][x] == current_player:
                from_idx = y * BOARD_SIZE + x
                
                # Check what moves are possible for this piece
                slide_moves = get_valid_slide_moves(board_2d, x, y)
                l_moves = get_valid_l_shape_moves(board_2d, x, y)
                valid_moves = slide_moves + l_moves
                
                if valid_moves:
                    from_mask[from_idx] = True
                    
                    # If this is the specific piece we want the to_mask for (Training)
                    if target_from_idx is not None and from_idx == target_from_idx:
                        for m in valid_moves:
                            to_idx = m["y"] * BOARD_SIZE + m["x"]
                            to_mask[to_idx] = True
                            
    return from_mask, to_mask


class ResBlock(nn.Module):
    """Residual block for deeper feature extraction."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)


class HorseRunPolicyNetV4(nn.Module):
    """
    Multi-Head CNN Policy Network for Horse Run Game.
    Input:  (batch, 3, 11, 11)
    Outputs:
        - logits_from: (batch, 121) — which piece to move
        - logits_to:   (batch, 121) — where to move it
    """
    def __init__(self, num_res_blocks=4, num_filters=128):
        super().__init__()
        
        # Shared Backbone
        self.initial_conv = nn.Sequential(
            nn.Conv2d(3, num_filters, 3, padding=1, bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True)
        )
        
        self.res_blocks = nn.Sequential(*[
            ResBlock(num_filters) for _ in range(num_res_blocks)
        ])
        
        # Head A: From
        self.head_from = nn.Sequential(
            nn.Conv2d(num_filters, 32, 1, bias=False), 
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * BOARD_SIZE * BOARD_SIZE, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, NUM_POSITIONS)  # 121 output classes
        )

        # Head B: To
        self.head_to = nn.Sequential(
            nn.Conv2d(num_filters, 32, 1, bias=False), 
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * BOARD_SIZE * BOARD_SIZE, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, NUM_POSITIONS)  # 121 output classes
        )
    
    def forward(self, x):
        common_features = self.initial_conv(x)
        common_features = self.res_blocks(common_features)
        
        logits_from = self.head_from(common_features)
        logits_to = self.head_to(common_features)
        
        return logits_from, logits_to


if __name__ == "__main__":
    model = HorseRunPolicyNetV4()
    dummy_input = torch.randn(2, 3, 11, 11)
    out_from, out_to = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"From output: {out_from.shape} (121)")
    print(f"To output:   {out_to.shape} (121)")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params:,}")
