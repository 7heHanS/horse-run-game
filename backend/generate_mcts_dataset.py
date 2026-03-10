"""
Overnight MCTS Dataset Generation Script.
Uses V4 Model + MCTS (3000/5000 sims) as a Teacher.
Saves winner-only moves with is_critical flagging.
"""
import os
import random
import csv
import time
import torch
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from engine import (
    BOARD_SIZE, check_win_condition, create_initial_board
)
from ai import get_all_possible_moves, simulate_move
from mcts import MCTS
from model_v4 import HorseRunPolicyNetV4

# Configuration
OUTPUT_FILE = "dataset_mcts_overnight.csv"
LOG_FILE = "generation_overnight.log"
TOTAL_GAMES = 2800
MAX_WORKERS = 24  # Leave some cores for the system

def log_message(msg):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

def run_mcts_game(game_id):
    """Process a single MCTS vs MCTS game."""
    try:
        # Each process needs its own model instance for thread-safety/GPU context
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = HorseRunPolicyNetV4().to(device)
        if os.path.exists("model_v4.pt"):
            model.load_state_dict(torch.load("model_v4.pt", map_location=device, weights_only=True))
        model.eval()

        # Decide simulation count for this game (mix 3000 and 5000)
        sims = 5000 if random.random() < 0.25 else 3000
        mcts_engine = MCTS(model, device, num_simulations=sims)

        board = create_initial_board()
        current_player = 1
        history = []
        turn_count = 0
        max_turns = 120

        while turn_count < max_turns:
            # Epsilon-greedy for data diversity
            # 5% chance to pick a random legal move
            possible_moves = get_all_possible_moves(board, current_player)
            if not possible_moves:
                break
                
            if random.random() < 0.05:
                best_move = random.choice(possible_moves)
            else:
                best_move = mcts_engine.search(board, current_player)
            
            if not best_move:
                break

            # Record state BEFORE applying move
            history.append({
                "board": [row[:] for row in board],
                "player": current_player,
                "from_c": best_move["pieceX"],
                "from_r": best_move["pieceY"],
                "to_c": best_move["targetX"],
                "to_r": best_move["targetY"],
            })

            board = simulate_move(board, best_move)
            if check_win_condition(best_move["targetX"], best_move["targetY"]):
                return history, current_player, sims

            current_player = 2 if current_player == 1 else 1
            turn_count += 1

        return history, 0, sims # Draw
    except Exception as e:
        log_message(f"Error in game {game_id}: {str(e)}")
        return [], 0, 0

def worker(game_id):
    history, winner, sims = run_mcts_game(game_id)
    if winner == 0:
        return []

    # Filter winner moves only
    winner_moves = [h for h in history if h["player"] == winner]
    total_w_moves = len(winner_moves)
    
    csv_rows = []
    for i, step in enumerate(winner_moves):
        # Final 10 moves of a winner are critical
        is_critical = 1 if (total_w_moves - i) <= 10 else 0
        
        row = []
        for r_row in step["board"]:
            row.extend(r_row)
        row.extend([
            step["player"],
            step["from_c"], step["from_r"],
            step["to_c"], step["to_r"],
            winner,
            is_critical
        ])
        csv_rows.append(row)
    
    return csv_rows

def main():
    log_message(f"Starting overnight generation: {TOTAL_GAMES} games, workers={MAX_WORKERS}")
    start_time = time.time()

    header = [f"cell_{y}_{x}" for y in range(BOARD_SIZE) for x in range(BOARD_SIZE)]
    header.extend([
        "player_turn",
        "action_from_c", "action_from_r",
        "action_to_c", "action_to_r",
        "winner", "is_critical"
    ])

    file_exists = os.path.exists(OUTPUT_FILE)
    total_records = 0
    games_done = 0

    with open(OUTPUT_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(worker, i) for i in range(TOTAL_GAMES)]
            for future in as_completed(futures):
                rows = future.result()
                if rows:
                    writer.writerows(rows)
                    total_records += len(rows)
                    f.flush()
                
                games_done += 1
                if games_done % 10 == 0:
                    elapsed = time.time() - start_time
                    avg = elapsed / games_done
                    eta = avg * (TOTAL_GAMES - games_done)
                    log_message(f"Progress: {games_done}/{TOTAL_GAMES} | Records: {total_records} | Elapsed: {elapsed/3600:.2f}h | ETA: {eta/3600:.2f}h")

    total_time = time.time() - start_time
    log_message(f"Finished! Total games: {games_done}, Records: {total_records}, Time: {total_time/3600:.2f}h")

if __name__ == "__main__":
    main()
