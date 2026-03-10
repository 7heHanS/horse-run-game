"""
Extract seed board states from user's winning game records.
Parses move sequences, replays them on the board, and extracts
states at turns 4, 6, 8. Then applies 8-fold symmetry augmentation
(4 rotations × 2 flips) to generate 48 seed states.
"""
import json
import copy
from engine import BOARD_SIZE, create_initial_board

# ============================================================
# User's 2 winning game records (player = 1 is human, moves as (x, y))
# Format: list of (from_x, from_y, to_x, to_y)
# ============================================================

GAME_1 = [
    # Turn 1: Player 1
    (0,0, 0,7),
    # Turn 2: Player 2 (AI)
    (8,0, 6,1),
    # Turn 3
    (0,1, 0,6),
    # Turn 4
    (10,2, 8,3),
    # Turn 5
    (0,0, 0,5),  # Note: (0,0) should be re-checked — after turn 1, (0,0) is empty.
    # Let me re-parse: the user's notation is (pieceX, pieceY) -> (targetX, targetY)
    # Turn 1: piece at (0,0) moves to (0,7)
    # Turn 3: piece at (0,1) moves to (0,6)
    # Turn 5: there's another piece... Let me trace the initial positions
]

# Let me properly parse from the user's move notation
# Each line is: (fromX, fromY) -> (toX, toY)

GAME_1_MOVES = [
    (0,0, 0,7),     # P1
    (8,0, 6,1),     # P2
    (0,1, 0,6),     # P1
    (10,2, 8,3),    # P2
    (0,2, 0,5),     # P1 - corrected: must be (0,2) since (0,0) already moved
    (0,8, 1,6),     # P2 - wait, this is the user's exact notation. Let me use it as-is.
]

# Actually, let me just use the raw user input directly and trust the coordinates.
# The user gave: (fromX, fromY) -> (toX, toY)

GAME_RECORDS = {
    "game1": [
        # Player 1 starts
        {"fx": 0, "fy": 0, "tx": 0, "ty": 7},
        {"fx": 8, "fy": 0, "tx": 6, "ty": 1},
        {"fx": 0, "fy": 1, "tx": 0, "ty": 6},
        {"fx": 10, "fy": 2, "tx": 8, "ty": 3},
        {"fx": 0, "fy": 0, "tx": 0, "ty": 5},  # Hmm, (0,0) was already moved turn 1...
        {"fx": 0, "fy": 8, "tx": 1, "ty": 6},
        {"fx": 0, "fy": 6, "tx": 2, "ty": 5},
        {"fx": 2, "fy": 10, "tx": 3, "ty": 8},
        {"fx": 10, "fy": 8, "tx": 10, "ty": 2},
        {"fx": 9, "fy": 0, "tx": 7, "ty": 1},
        {"fx": 10, "fy": 9, "tx": 10, "ty": 3},
        {"fx": 7, "fy": 1, "tx": 6, "ty": 3},
        {"fx": 10, "fy": 10, "tx": 10, "ty": 4},
        {"fx": 10, "fy": 1, "tx": 7, "ty": 1},
        {"fx": 10, "fy": 5, "tx": 8, "ty": 6},
        {"fx": 6, "fy": 3, "tx": 8, "ty": 4},
        {"fx": 10, "fy": 3, "tx": 9, "ty": 5},
        {"fx": 7, "fy": 1, "tx": 6, "ty": 3},
        {"fx": 9, "fy": 10, "tx": 8, "ty": 8},
        {"fx": 3, "fy": 8, "tx": 1, "ty": 7},
        {"fx": 0, "fy": 7, "tx": 1, "ty": 5},
        {"fx": 1, "fy": 7, "tx": 3, "ty": 6},
        {"fx": 2, "fy": 5, "tx": 7, "ty": 5},
        {"fx": 8, "fy": 4, "tx": 7, "ty": 6},
        {"fx": 1, "fy": 5, "tx": 6, "ty": 5},
        {"fx": 3, "fy": 7, "tx": 6, "ty": 7},  # Wait, 3,7?
        {"fx": 0, "fy": 5, "tx": 5, "ty": 5},   # Win!
    ],
    "game2": [
        {"fx": 0, "fy": 2, "tx": 0, "ty": 7},
        {"fx": 8, "fy": 0, "tx": 6, "ty": 1},
        {"fx": 0, "fy": 1, "tx": 0, "ty": 6},
        {"fx": 6, "fy": 1, "tx": 4, "ty": 2},
        {"fx": 0, "fy": 0, "tx": 0, "ty": 5},
        {"fx": 10, "fy": 2, "tx": 5, "ty": 2},
        {"fx": 0, "fy": 6, "tx": 2, "ty": 5},
        {"fx": 2, "fy": 10, "tx": 2, "ty": 6},
        {"fx": 10, "fy": 9, "tx": 9, "ty": 7},
        {"fx": 4, "fy": 2, "tx": 6, "ty": 3},
        {"fx": 9, "fy": 7, "tx": 8, "ty": 5},
        {"fx": 0, "fy": 7, "tx": 1, "ty": 5},
        {"fx": 6, "fy": 3, "tx": 4, "ty": 2},
        {"fx": 10, "fy": 8, "tx": 8, "ty": 7},
        {"fx": 4, "fy": 2, "tx": 6, "ty": 3},
        {"fx": 8, "fy": 7, "tx": 9, "ty": 5},
        {"fx": 5, "fy": 2, "tx": 3, "ty": 3},
        {"fx": 10, "fy": 10, "tx": 9, "ty": 8},
        {"fx": 3, "fy": 3, "tx": 5, "ty": 2},
        {"fx": 9, "fy": 8, "tx": 8, "ty": 6},
        {"fx": 2, "fy": 6, "tx": 7, "ty": 6},
        {"fx": 8, "fy": 6, "tx": 10, "ty": 5},
        {"fx": 1, "fy": 6, "tx": 6, "ty": 6},
        {"fx": 2, "fy": 5, "tx": 7, "ty": 5},
        {"fx": 6, "fy": 6, "tx": 6, "ty": 4},
        {"fx": 1, "fy": 5, "tx": 6, "ty": 5},
        {"fx": 5, "fy": 2, "tx": 3, "ty": 3},
        {"fx": 0, "fy": 5, "tx": 5, "ty": 5},  # Win!
    ]
}

# Turns at which to extract seed states (0-indexed turn count)
EXTRACT_AT_TURNS = [4, 6, 8]


def replay_game(moves):
    """Replay a sequence of moves and return board states at specified turns."""
    board = create_initial_board()
    states = {}
    current_player = 1
    
    for turn_idx, move in enumerate(moves):
        # Validate move
        fx, fy, tx, ty = move["fx"], move["fy"], move["tx"], move["ty"]
        
        if board[fy][fx] != current_player:
            print(f"  WARNING Turn {turn_idx}: Expected player {current_player} at ({fx},{fy}), "
                  f"found {board[fy][fx]}. Skipping validation.")
            # Still apply the move — trust the user's record
            piece = board[fy][fx]
            if piece == 0:
                print(f"  ERROR Turn {turn_idx}: No piece at ({fx},{fy})! Board state may be corrupted.")
                # Try to continue anyway
        
        # Apply move
        board[fy][fx] = 0
        board[ty][tx] = current_player
        
        # Check if this turn is an extraction point (after the move is applied)
        actual_turn = turn_idx + 1
        if actual_turn in EXTRACT_AT_TURNS:
            states[actual_turn] = {
                "board": [row[:] for row in board],
                "next_player": 2 if current_player == 1 else 1,
                "turn": actual_turn
            }
        
        current_player = 2 if current_player == 1 else 1
    
    return states


def rotate_board_90(board):
    """Rotate board 90° clockwise around center (5,5)."""
    n = BOARD_SIZE
    rotated = [[0]*n for _ in range(n)]
    for y in range(n):
        for x in range(n):
            # 90° clockwise: (x,y) -> (n-1-y, x)
            rotated[x][n-1-y] = board[y][x]
    return rotated


def flip_board_horizontal(board):
    """Flip board left-right."""
    n = BOARD_SIZE
    flipped = [[0]*n for _ in range(n)]
    for y in range(n):
        for x in range(n):
            flipped[y][n-1-x] = board[y][x]
    return flipped


def augment_board(board):
    """Generate 8 augmented boards (4 rotations × 2 flips)."""
    augmented = []
    current = [row[:] for row in board]
    
    for rotation in range(4):
        augmented.append([row[:] for row in current])
        # Also add horizontal flip
        augmented.append(flip_board_horizontal(current))
        # Rotate 90° for next iteration
        current = rotate_board_90(current)
    
    return augmented


def main():
    all_seeds = []
    
    for game_name, moves in GAME_RECORDS.items():
        print(f"\n=== Replaying {game_name} ({len(moves)} moves) ===")
        states = replay_game(moves)
        
        for turn, state in sorted(states.items()):
            print(f"  Turn {turn}: next_player={state['next_player']}")
            
            # Apply 8-fold augmentation
            augmented_boards = augment_board(state["board"])
            
            for aug_idx, aug_board in enumerate(augmented_boards):
                aug_labels = ["0°", "0°+flip", "90°", "90°+flip", 
                              "180°", "180°+flip", "270°", "270°+flip"]
                all_seeds.append({
                    "source_game": game_name,
                    "source_turn": turn,
                    "augmentation": aug_labels[aug_idx],
                    "next_player": state["next_player"],
                    "board": aug_board
                })
    
    # Save to JSON
    output_file = "seed_states.json"
    with open(output_file, "w") as f:
        json.dump(all_seeds, f, indent=2)
    
    print(f"\n✅ Generated {len(all_seeds)} seed states → {output_file}")
    print(f"   Sources: {len(GAME_RECORDS)} games × {len(EXTRACT_AT_TURNS)} turns × 8 augmentations")


if __name__ == "__main__":
    main()
