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
    from train import AIPolicyNet
except ImportError:
    pass # In case dependencies are still installing

WEIGHTS = {
    "WIN": 100000,
    "MEADOW": 500,
    "SETUP_THREAT": 1500,
    "BLOCKING": 1000,
    "CENTER_CONTROL": 10,
    "MOBILITY": 1
}

def get_ml_best_move(board: list[list[int]], current_player: int, model_path="model.pt"):
    """
    Predicts the best move using the trained neural network model.
    """
    if not os.path.exists(model_path):
        return None
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AIPolicyNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Flatten board and convert to tensor
    flat_board = []
    for row in board:
         flat_board.extend(row)
         
    features = flat_board + [current_player]
    input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        
    # Get top predicted coordinate
    # Use softmax to get probabilities
    probs = torch.nn.functional.softmax(outputs[0], dim=0)
    
    # We predicted the target destination (to_r * BOARD_SIZE + to_c)
    # Get all valid possible moves first
    possible_moves = get_all_possible_moves(board, current_player)
    if not possible_moves:
        return None
        
    # Score each possible move based on the network's probability output for its target destination
    best_move = None
    best_prob = -1
    
    for move in possible_moves:
        target_idx = move["targetY"] * BOARD_SIZE + move["targetX"]
        prob = probs[target_idx].item()
        
        if prob > best_prob:
            best_prob = prob
            best_move = move
            
    return best_move

def find_best_move(board: list[list[int]], ai_player: int, human_player: int, depth: int, use_ml=False):
    if use_ml:
        ml_move = get_ml_best_move(board, ai_player)
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
