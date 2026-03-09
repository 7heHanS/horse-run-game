"""
Horse Run Game - AI Policy Network V3
CNN-based architecture with 3-channel spatial input and from-to action space.
"""
import torch
import torch.nn as nn

BOARD_SIZE = 11
NUM_POSITIONS = BOARD_SIZE * BOARD_SIZE  # 121
NUM_ACTIONS = NUM_POSITIONS * NUM_POSITIONS  # 14641 (from_idx * 121 + to_idx)

# Special positions on the board
OASIS = (5, 5)
MEADOW_SET = {
    (5, 4), (5, 6), (4, 5), (6, 5),
    (5, 3), (5, 7), (3, 5), (7, 5),
    (4, 4), (6, 4), (4, 6), (6, 6)
}

def board_to_channels(board, current_player):
    """
    Converts a flat or 2D board + player into a (3, 11, 11) tensor.
    Channel 0: Current player's pieces (1 where piece exists, 0 otherwise)
    Channel 1: Opponent's pieces (1 where piece exists, 0 otherwise)
    Channel 2: Special tiles (Oasis=1.0, Meadow=0.5, Desert=0.0)
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
            
            # Special tiles channel
            if (x, y) == OASIS:
                channels[2, y, x] = 1.0
            elif (x, y) in MEADOW_SET:
                channels[2, y, x] = 0.5
    
    return channels


def get_legal_move_mask(board_2d, current_player):
    """
    Returns a (14641,) boolean tensor where True = legal move.
    Uses the game engine's move generation to identify all valid (from, to) pairs.
    """
    from engine import get_valid_slide_moves, get_valid_l_shape_moves
    
    mask = torch.zeros(NUM_ACTIONS, dtype=torch.bool)
    
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            if board_2d[y][x] == current_player:
                from_idx = y * BOARD_SIZE + x
                
                # Get all valid destinations for this piece
                slide_moves = get_valid_slide_moves(board_2d, x, y)
                l_moves = get_valid_l_shape_moves(board_2d, x, y)
                
                for m in slide_moves + l_moves:
                    to_idx = m["y"] * BOARD_SIZE + m["x"]
                    action_idx = from_idx * NUM_POSITIONS + to_idx
                    mask[action_idx] = True
    
    return mask



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


class HorseRunPolicyNetV3(nn.Module):
    """
    CNN Policy Network for Horse Run Game.
    Input:  (batch, 3, 11, 11)
    Output: (batch, 14641) — probability over (from_position, to_position) pairs
    """
    def __init__(self, num_res_blocks=4, num_filters=128):
        super().__init__()
        
        # Initial convolution: 3 channels -> num_filters
        self.initial_conv = nn.Sequential(
            nn.Conv2d(3, num_filters, 3, padding=1, bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True)
        )
        
        # Residual tower
        self.res_blocks = nn.Sequential(*[
            ResBlock(num_filters) for _ in range(num_res_blocks)
        ])
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_filters, 32, 1, bias=False),  # 1x1 conv to reduce channels
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * BOARD_SIZE * BOARD_SIZE, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, NUM_ACTIONS)  # 14641 output classes
        )
    
    def forward(self, x):
        x = self.initial_conv(x)
        x = self.res_blocks(x)
        return self.policy_head(x)


if __name__ == "__main__":
    # Quick test
    model = HorseRunPolicyNetV3()
    dummy_input = torch.randn(2, 3, 11, 11)
    output = model(dummy_input)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Action space: {NUM_ACTIONS} (121 from × 121 to)")
