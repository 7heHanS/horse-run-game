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

def get_ml_best_move(board: list[list[int]], current_player: int, model_path="model_v4.pt", use_mcts=False, mcts_simulations=100):
    """
    Predicts the best move using the trained neural network model.
    Supports V3 (CNN, from-to) and legacy V2/V1 (MLP, to-only).
    """
    global _cached_model, _cached_model_path

    if not os.path.exists(model_path):
        # Try legacy model path
        if os.path.exists("model.pt"):
            model_path = "model.pt"
        else:
            return None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model (with caching)
    if _cached_model is None or _cached_model_path != model_path:
        if ML_VERSION == 4 and "v4" in model_path:
            model = PolicyNetV4().to(device)
        elif ML_VERSION == 3 and "v3" in model_path:
            model = HorseRunPolicyNetV3().to(device)
        elif ML_VERSION >= 2:
            model = AIPolicyNet().to(device)
        else:
            return None
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
        _cached_model = model
        _cached_model_path = model_path
    else:
        model = _cached_model

    possible_moves = get_all_possible_moves(board, current_player)
    if not possible_moves:
        return None

    # V4 MCTS Integration
    if use_mcts and ML_VERSION == 4 and "v4" in model_path:
        from mcts import MCTS
        mcts_engine = MCTS(model, device, num_simulations=mcts_simulations)
        return mcts_engine.search(board, current_player)

    # V4: Multi-Head CNN (From/To) + Legal Move Masking (Greedy)
    if ML_VERSION == 4 and "v4" in model_path:
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
    if ML_VERSION == 3 and "v3" in model_path:
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


def find_best_move(board: list[list[int]], ai_player: int, human_player: int, depth: int, use_ml=False, use_mcts=False, mcts_simulations=100):
    if use_ml:
        ml_move = get_ml_best_move(board, ai_player, use_mcts=use_mcts, mcts_simulations=mcts_simulations)
        if ml_move:
           return ml_move
        print("ML Model not found or failed, falling back to Minimax.")
        
    best_score = -float('inf')
    best_move = None
    
    possible_moves = get_all_possible_moves(board, ai_player)
    
    for move in possible_moves:
        next_board = simulate_move(board, move)
        if check_win_condition(move["targetX"], move["targetY"]):
            return move
            
        score = minimax(next_board, depth - 1, -float('inf'), float('inf'), False, ai_player, human_player)
        
        if score > best_score:
            best_score = score
            best_move = move
            
    return best_move

def minimax(board: list[list[int]], depth: int, alpha: float, beta: float, is_maximizing: bool, ai_player: int, human_player: int):
    if depth == 0:
        return evaluate_board(board, ai_player, human_player)
        
    current_player = ai_player if is_maximizing else human_player
    possible_moves = get_all_possible_moves(board, current_player)
    
    if len(possible_moves) == 0:
        return evaluate_board(board, ai_player, human_player)
        
    if is_maximizing:
        max_eval = -float('inf')
        for move in possible_moves:
            next_board = simulate_move(board, move)
            if check_win_condition(move["targetX"], move["targetY"]):
                return WEIGHTS["WIN"] + depth
                
            eval_score = minimax(next_board, depth - 1, alpha, beta, False, ai_player, human_player)
            max_eval = max(max_eval, eval_score)
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float('inf')
        for move in possible_moves:
            next_board = simulate_move(board, move)
            if check_win_condition(move["targetX"], move["targetY"]):
                return -WEIGHTS["WIN"] - depth
                
            eval_score = minimax(next_board, depth - 1, alpha, beta, True, ai_player, human_player)
            min_eval = min(min_eval, eval_score)
            beta = min(beta, eval_score)
            if beta <= alpha:
                break
        return min_eval

def get_all_possible_moves(board: list[list[int]], player: int):
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
    def dist_to_center(m):
        return abs(m["targetX"] - 5) + abs(m["targetY"] - 5)
        
    moves.sort(key=dist_to_center)
    return moves

def simulate_move(board: list[list[int]], move: dict):
    next_board = [row[:] for row in board]
    player = next_board[move["pieceY"]][move["pieceX"]]
    next_board[move["pieceY"]][move["pieceX"]] = 0
    next_board[move["targetY"]][move["targetX"]] = player
    return next_board

def evaluate_board(board: list[list[int]], ai_player: int, human_player: int):
    score = 0
    
    # 1. Center Control
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            p = board[y][x]
            if p != 0:
                dist_to_center = abs(x - 5) + abs(y - 5)
                center_score = (10 - dist_to_center) * WEIGHTS["CENTER_CONTROL"]
                if p == ai_player:
                    score += center_score
                elif p == human_player:
                    score -= center_score
                    
    # 2. Meadow control and Setup
    for m in MEADOW_POSITIONS:
        stopper_piece = board[m["y"]][m["x"]]
        if stopper_piece != 0:
            owner = stopper_piece
            opponent = human_player if owner == ai_player else ai_player
            sign = 1 if owner == ai_player else -1
            
            score += WEIGHTS["MEADOW"] * sign
            
            # 3. Launchers / Blockers
            is_horizontal = (m["y"] == 5)
            is_vertical = (m["x"] == 5)
            
            owner_launchers = 0
            opponent_blockers = 0
            
            if is_horizontal:
                for x in range(BOARD_SIZE):
                    if x == m["x"] or x == 5:
                        continue
                    p = board[m["y"]][x]
                    if p != 0:
                        if p == owner: owner_launchers += 1
                        if p == opponent: opponent_blockers += 1
            elif is_vertical:
                for y in range(BOARD_SIZE):
                    if y == m["y"] or y == 5:
                        continue
                    p = board[y][m["x"]]
                    if p != 0:
                        if p == owner: owner_launchers += 1
                        if p == opponent: opponent_blockers += 1
                        
            if owner_launchers > 0:
                if opponent_blockers == 0:
                    score += WEIGHTS["SETUP_THREAT"] * sign
                else:
                    score += WEIGHTS["BLOCKING"] * (-sign)
                    
    return score
