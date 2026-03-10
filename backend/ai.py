import math
import random
import os
from engine import (
    BOARD_SIZE, OASIS_POS, MEADOW_POSITIONS,
    get_valid_slide_moves, get_valid_l_shape_moves, check_win_condition
)

try:
    import torch
    import numpy as np
    # Try V4 first, then down the versions
    try:
        from model_v4 import HorseRunPolicyNetV4 as PolicyNetV4, board_to_channels, get_v4_masks, NUM_POSITIONS
        ML_VERSION = 4
    except ImportError:
        try:
            from model_v3 import HorseRunPolicyNetV3, board_to_channels, get_legal_move_mask, NUM_POSITIONS
            ML_VERSION = 3
        except ImportError:
            try:
                from train_model_v2 import AIPolicyNetV2 as AIPolicyNet
                ML_VERSION = 2
            except ImportError:
                from train import AIPolicyNet
                ML_VERSION = 1
except ImportError:
    ML_VERSION = 0

WEIGHTS = {
    "WIN": 100000,
    "MEADOW": 500,
    "SETUP_THREAT": 1500,
    "BLOCKING": 1000,
    "CENTER_CONTROL": 10,
    "MOBILITY": 1
}

# Cache model to avoid reloading on every call
_cached_model = None
_cached_model_path = None

def get_all_possible_moves(board: list[list[int]], player: int):
    """Enumerate all legal moves for a player. Used by ML inference pipeline."""
    moves = []
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            if board[y][x] == player:
                slide_moves = get_valid_slide_moves(board, x, y)
                l_shape_moves = get_valid_l_shape_moves(board, x, y)
                
                for m in slide_moves + l_shape_moves:
                    moves.append({
                        "pieceX": x, "pieceY": y,
                        "targetX": m["x"], "targetY": m["y"],
                        "type": m["type"], "player": player
                    })
    # Sort moves by distance to oasis (center)
    moves.sort(key=lambda m: abs(m["targetX"] - 5) + abs(m["targetY"] - 5))
    return moves

def simulate_move(board: list[list[int]], move: dict) -> list[list[int]]:
    """Apply a move to a board and return the new board state."""
    new_board = [row[:] for row in board]
    player = new_board[move["pieceY"]][move["pieceX"]]
    new_board[move["pieceY"]][move["pieceX"]] = 0
    new_board[move["targetY"]][move["targetX"]] = player
    return new_board

def evaluate_board(board: list[list[int]], ai_player: int, human_player: int) -> float:
    """Heuristic board evaluation for MCTS leaf nodes."""
    score = 0.0
    
    # 1. Coordinate-based scores & Stopper detection
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            p = board[y][x]
            if p == 0: continue
            
            dist = abs(x - 5) + abs(y - 5)
            # Base center control score
            center_score = (10 - dist) * WEIGHTS["CENTER_CONTROL"]
            
            # Proximity threat: extra weights for being very close to winning
            threat_score = 0
            if dist <= 8:
                threat_score = (10 - dist) * 2000  # Massive progressive penalty

            if p == ai_player:
                score += (center_score + threat_score)
            elif p == human_player:
                score -= (center_score + threat_score)

    # 2. Axis Patrol & Stopper Detection
    stoppers = [
        (6, 5, "L"), (4, 5, "R"), (5, 6, "T"), (5, 4, "B")
    ]
    
    # Axis counts for cluster detection
    ai_on_x5, human_on_x5 = 0, 0
    ai_on_y5, human_on_y5 = 0, 0
    
    for i in range(BOARD_SIZE):
        if board[5][i] == ai_player: ai_on_x5 += 1
        elif board[5][i] == human_player: human_on_x5 += 1
        
        if board[i][5] == ai_player: ai_on_y5 += 1
        elif board[i][5] == human_player: human_on_y5 += 1

    # Axis Patrol Reward: AI pieces on x=5 or y=5 are good defenders
    score += (ai_on_x5 + ai_on_y5) * 2000
    score -= (human_on_x5 + human_on_y5) * 3000 # Opponent on axis is VERY bad

    # Cluster Penalty: If opponent has >1 piece on an axis, it's a huge setup threat
    if human_on_x5 > 1: score -= 10000
    if human_on_y5 > 1: score -= 10000

    for sx, sy, side in stoppers:
        # Check for pieces at stopper positions (SX, SY)
        p_at_stopper = board[sy][sx]
        
        for player in [1, 2]:
            is_imminent = False
            is_blocked_threat = False
            
            # Check for pieces that could slide into (5,5)
            if side == "L": # Slide RIGHT to (5,5), blocked by (6,5)
                sources = [x for x in range(5) if board[5][x] == player]
                for x in sources:
                    if all(board[5][i] == 0 for i in range(x+1, 5)):
                        is_imminent = True; break
                    else: is_blocked_threat = True
            elif side == "R": # Slide LEFT to (5,5), blocked by (4,5)
                sources = [x for x in range(6, 11) if board[5][x] == player]
                for x in sources:
                    if all(board[5][i] == 0 for i in range(5+1, x)):
                        is_imminent = True; break
                    else: is_blocked_threat = True
            elif side == "T": # Slide DOWN to (5,5), blocked by (5,6)
                sources = [y for y in range(5) if board[y][5] == player]
                for y in sources:
                    if all(board[i][5] == 0 for i in range(y+1, 5)):
                        is_imminent = True; break
                    else: is_blocked_threat = True
            elif side == "B": # Slide UP to (5,5), blocked by (5,4)
                sources = [y for y in range(6, 11) if board[y][5] == player]
                for y in sources:
                    if all(board[i][5] == 0 for i in range(5+1, y)):
                        is_imminent = True; break
                    else: is_blocked_threat = True
            
            if is_imminent and p_at_stopper != 0:
                threat_level = 50000 
            elif is_imminent and p_at_stopper == 0:
                threat_level = 15000 # Increased from 10k
            elif is_blocked_threat and p_at_stopper != 0:
                threat_level = 8000  # Increased from 5k
            else:
                threat_level = 0
                
            if player == ai_player: score += threat_level
            else: score -= threat_level

    # 3. Mobility
    ai_moves = len(get_all_possible_moves(board, ai_player))
    human_moves = len(get_all_possible_moves(board, human_player))
    score += (ai_moves - human_moves) * WEIGHTS["MOBILITY"]
    return score

def get_model():
    """Loads and caches the ML model."""
    global _cached_model, _cached_model_path
    
    model_path = "model_v6.pt"
    
    if not os.path.exists(model_path):
        return None, None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model (with caching)
    if _cached_model is None or _cached_model_path != model_path:
        if ML_VERSION == 4 and ("v4" in model_path or "v5" in model_path or "v6" in model_path):
            model = PolicyNetV4().to(device)
        elif ML_VERSION == 3 and "v3" in model_path:
            model = HorseRunPolicyNetV3().to(device)
        elif ML_VERSION >= 2:
            model = AIPolicyNet().to(device)
        else:
            return None, None
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
        _cached_model = model
        _cached_model_path = model_path
    else:
        model = _cached_model
        
    return model, device

def get_ml_best_move(board: list[list[int]], current_player: int):
    """
    Predicts the best move using the trained neural network model (V5/V4).
    """
    model, device = get_model()
    if model is None:
        return None

    possible_moves = get_all_possible_moves(board, current_player)
    if not possible_moves:
        return None

    # V4: Multi-Head CNN (From/To) + Legal Move Masking (Greedy)
    if ML_VERSION == 4:
        channels = board_to_channels(board, current_player).unsqueeze(0).to(device)
        
        # Get just the from_mask for inference
        from_mask, _ = get_v4_masks(board, current_player)
        from_mask = from_mask.to(device)

        with torch.no_grad():
            logits_from, logits_to = model(channels)
            
            # Mask illegal 'from' pieces
            logits_from[0][~from_mask] = -1e9
            
            probs_from = torch.nn.functional.softmax(logits_from[0], dim=0)
            probs_to = torch.nn.functional.softmax(logits_to[0], dim=0)

        best_move = None
        best_prob = -1

        for move in possible_moves:
            from_idx = move["pieceY"] * BOARD_SIZE + move["pieceX"]
            to_idx = move["targetY"] * BOARD_SIZE + move["targetX"]
            
            # P(move) = P(from) * P(to)
            prob = probs_from[from_idx].item() * probs_to[to_idx].item()

            if prob > best_prob:
                best_prob = prob
                best_move = move

        return best_move

    # V3: CNN with from-to action space + Legal Move Masking
    if ML_VERSION == 3:
        channels = board_to_channels(board, current_player).unsqueeze(0).to(device)

        # Generate legal move mask
        legal_mask = get_legal_move_mask(board, current_player).to(device)

        with torch.no_grad():
            logits = model(channels)
            # Apply legal move masking: set illegal logits to -inf
            logits[0][~legal_mask] = -1e9
            probs = torch.nn.functional.softmax(logits[0], dim=0)

        best_move = None
        best_prob = -1

        for move in possible_moves:
            from_idx = move["pieceY"] * BOARD_SIZE + move["pieceX"]
            to_idx = move["targetY"] * BOARD_SIZE + move["targetX"]
            action_idx = from_idx * NUM_POSITIONS + to_idx
            prob = probs[action_idx].item()

            if prob > best_prob:
                best_prob = prob
                best_move = move

        return best_move

    # Legacy V2/V1: MLP with to-only action space
    else:
        flat_board = []
        for row in board:
            flat_board.extend(row)
        features = flat_board + [current_player]
        input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs[0], dim=0)

        best_move = None
        best_prob = -1
        for move in possible_moves:
            target_idx = move["targetY"] * BOARD_SIZE + move["targetX"]
            prob = probs[target_idx].item()
            if prob > best_prob:
                best_prob = prob
                best_move = move

        return best_move

def find_best_move(board: list[list[int]], ai_player: int, human_player: int, depth: int, use_ml=True, use_mcts=False, mcts_simulations=100):
    """
    Since Phase 13, the backend only serves Deep Learning (V5) predictions. 
    Minimax is handled client-side in JS.
    """
    if use_mcts:
        from mcts import MCTS
        model, device = get_model()
        if model is None:
            return get_ml_best_move(board, ai_player)
        mcts = MCTS(model=model, device=device, num_simulations=mcts_simulations)
        return mcts.search(board, ai_player)
    else:
        return get_ml_best_move(board, ai_player)
