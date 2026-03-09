import os
import random
import json
from engine import BOARD_SIZE, check_win_condition
from ai import find_best_move, get_all_possible_moves, simulate_move

# 게임 초기 배치 세팅
INITIAL_POSITIONS_P1 = [
    {"x": 10, "y": 8}, {"x": 10, "y": 9}, {"x": 10, "y": 10}, {"x": 9, "y": 10}, {"x": 8, "y": 10},
    {"x": 0, "y": 0}, {"x": 1, "y": 0}, {"x": 2, "y": 0}, {"x": 0, "y": 1}, {"x": 0, "y": 2}
]

INITIAL_POSITIONS_P2 = [
    {"x": 0, "y": 8}, {"x": 0, "y": 9}, {"x": 0, "y": 10}, {"x": 1, "y": 10}, {"x": 2, "y": 10},
    {"x": 10, "y": 0}, {"x": 9, "y": 0}, {"x": 8, "y": 0}, {"x": 10, "y": 1}, {"x": 10, "y": 2}
]

def create_initial_board():
    board = [[0 for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    for pos in INITIAL_POSITIONS_P1:
        board[pos["y"]][pos["x"]] = 1
    for pos in INITIAL_POSITIONS_P2:
        board[pos["y"]][pos["x"]] = 2
    return board

def run_self_play_game(epsilon=0.1, depth=2):
    """
    Minimax vs Minimax self-play simulator.
    `epsilon`: epsilon-greedy parameter. Chance to make a random valid move to add variance.
    Returns a list of (state, action, player, reward) tuples for this game.
    """
    board = create_initial_board()
    current_player = 1
    
    # Store game history as tuples: (board_state, pieceX, pieceY, targetX, targetY, player)
    history = []
    
    turn_count = 0
    max_turns = 150 # Prevent infinite loops
    
    # Track the last few stringified board states to avoid immediate 2-move repetition loops
    visited_states = []
    
    while turn_count < max_turns:
        human_player = 1 if current_player == 2 else 2
        
        possible_moves = get_all_possible_moves(board, current_player)
        
        # Filter moves that lead to a recently visited state (to prevent infinite loops)
        valid_non_repeating_moves = []
        for m in possible_moves:
            next_b = simulate_move(board, m)
            if str(next_b) not in visited_states[-4:]: # Check last 4 states
                valid_non_repeating_moves.append(m)
                
        # Fallback to all moves if we are trapped
        if not valid_non_repeating_moves:
            valid_non_repeating_moves = possible_moves
            
        if not valid_non_repeating_moves:
            break
            
        # Determine move
        if random.random() < epsilon:
            # Random valid move for variance
            best_move = random.choice(valid_non_repeating_moves)
        else:
            # We override the minimax `possible_moves` inside `find_best_move` by 
            # sorting and picking the top move from `valid_non_repeating_moves`
            best_score = -float('inf')
            best_move = None
            
            for move in valid_non_repeating_moves:
                next_board = simulate_move(board, move)
                if check_win_condition(move["targetX"], move["targetY"]):
                    best_move = move
                    break
                    
                from ai import minimax
                score = minimax(next_board, depth - 1, -float('inf'), float('inf'), False, current_player, human_player)
                
                if score > best_score:
                    best_score = score
                    best_move = move
                    
            if not best_move:
                 best_move = valid_non_repeating_moves[0]
                
        # Save state and action
        state_copy = [row[:] for row in board]
        history.append({
            "board": state_copy,
            "player": current_player,
            "action": {
                "from_c": best_move["pieceX"],
                "from_r": best_move["pieceY"],
                "to_c": best_move["targetX"],
                "to_r": best_move["targetY"]
            }
        })
        
        # Apply move
        board = simulate_move(board, best_move)
        visited_states.append(str(board))
        
        # Check win condition
        if check_win_condition(best_move["targetX"], best_move["targetY"]):
            print(f"  Game ended at turn {turn_count}! Winner: Player {current_player}")
            return history, current_player
            
        current_player = human_player
        turn_count += 1
        print(f"  Play turn {turn_count} done.")
        
    print(f"  Game ended in draw at turn {max_turns}.")
    return history, 0 # 0 means draw or max turns reached

def generate_dataset(num_games=1000, output_file="dataset.csv"):
    all_game_data = []
    
    print(f"Starting self-play simulation for {num_games} games...")
    
    for i in range(num_games):
        # Reduce depth to speed up data generation
        depth = random.choice([1, 2]) 
        epsilon = random.uniform(0.1, 0.4)
        
        history, winner = run_self_play_game(epsilon=epsilon, depth=depth)
        
        if winner != 0:
            for step in history:
                step["winner"] = winner
            all_game_data.extend(history)
            
        if (i + 1) % 50 == 0:
            print(f"Completed {i + 1}/{num_games} games...")
            
    print(f"Simulation complete. Generated {len(all_game_data)} state-action pairs.")
    
    import csv
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header
        header = [f"cell_{y}_{x}" for y in range(BOARD_SIZE) for x in range(BOARD_SIZE)]
        header.extend(["player_turn", "action_from_c", "action_from_r", "action_to_c", "action_to_r", "winner"])
        writer.writerow(header)
        
        for data in all_game_data:
            row = []
            # Flatten board
            for r in data["board"]:
                row.extend(r)
            row.append(data["player"])
            row.append(data["action"]["from_c"])
            row.append(data["action"]["from_r"])
            row.append(data["action"]["to_c"])
            row.append(data["action"]["to_r"])
            row.append(data["winner"])
            
            writer.writerow(row)
            
    print(f"Dataset CSV saved to {output_file}")

if __name__ == "__main__":
    generate_dataset(num_games=500) # Output 500 games for an initial learning set
