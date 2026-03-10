"""
Targeted MCTS Dataset Generation.
Loads seed board states from seed_states.json and runs V5+MCTS 1500 self-play
from those positions. Generates defense training data against side-column attacks.
"""
import os
import json
import random
import csv
import time
import torch
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from engine import BOARD_SIZE, check_win_condition
from ai import get_all_possible_moves
from mcts import MCTS
from model_v4 import HorseRunPolicyNetV4

# Configuration
SEED_FILE = "seed_states.json"
OUTPUT_FILE = "dataset_targeted_side.csv"
LOG_FILE = "generation_targeted.log"
GAMES_PER_SEED = 21       # 48 seeds × 21 = 1,008 games
MCTS_SIMULATIONS = 1500   # Reduced from 3000 — mid-game seeds have smaller search space
MAX_WORKERS = 28           # 32 cores - 4 reserved for user (web browsing)
MAX_TURNS_PER_GAME = 80   # Shorter than full-game since we start mid-game
MODEL_PATH = "model_v5.pt"

def log_message(msg):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

def simulate_move(board, move):
    """Apply a move to a board and return the new board state."""
    new_board = [row[:] for row in board]
    player = new_board[move["pieceY"]][move["pieceX"]]
    new_board[move["pieceY"]][move["pieceX"]] = 0
    new_board[move["targetY"]][move["targetX"]] = player
    return new_board

def run_targeted_game(seed_state, game_id):
    """Run a single MCTS vs MCTS game starting from a seed board state."""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = HorseRunPolicyNetV4().to(device)
        if os.path.exists(MODEL_PATH):
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
        model.eval()

        mcts_engine = MCTS(model, device, num_simulations=MCTS_SIMULATIONS)

        board = [row[:] for row in seed_state["board"]]
        current_player = seed_state["next_player"]
        history = []
        turn_count = 0

        while turn_count < MAX_TURNS_PER_GAME:
            possible_moves = get_all_possible_moves(board, current_player)
            if not possible_moves:
                break

            # 5% epsilon-greedy for data diversity
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
                return history, current_player

            current_player = 2 if current_player == 1 else 1
            turn_count += 1

        return history, 0  # Draw
    except Exception as e:
        log_message(f"Error in game {game_id}: {str(e)}")
        return [], 0

def worker(args):
    """Worker function for parallel execution."""
    seed_idx, seed_state, local_game_id, global_game_id = args
    history, winner = run_targeted_game(seed_state, global_game_id)
    
    if winner == 0:
        return [], seed_idx, global_game_id

    # Filter winner moves only
    winner_moves = [h for h in history if h["player"] == winner]
    total_w_moves = len(winner_moves)

    csv_rows = []
    for i, step in enumerate(winner_moves):
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

    return csv_rows, seed_idx, global_game_id

def main():
    # Load seeds
    if not os.path.exists(SEED_FILE):
        print(f"Error: {SEED_FILE} not found. Run extract_seed_states.py first.")
        return

    with open(SEED_FILE) as f:
        seeds = json.load(f)

    total_seeds = len(seeds)
    total_games = total_seeds * GAMES_PER_SEED

    log_message(f"=== Targeted MCTS Dataset Generation ===")
    log_message(f"Seeds: {total_seeds}, Games/Seed: {GAMES_PER_SEED}, Total: {total_games}")
    log_message(f"MCTS Simulations: {MCTS_SIMULATIONS}, Workers: {MAX_WORKERS}")
    log_message(f"Model: {MODEL_PATH}")

    start_time = time.time()

    # Prepare work items
    work_items = []
    global_id = 0
    for seed_idx, seed in enumerate(seeds):
        for local_id in range(GAMES_PER_SEED):
            work_items.append((seed_idx, seed, local_id, global_id))
            global_id += 1

    # CSV Header
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
    games_with_winner = 0

    with open(OUTPUT_FILE, 'a', newline='') as f:
        csv_writer = csv.writer(f)
        if not file_exists:
            csv_writer.writerow(header)

        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(worker, item) for item in work_items]
            
            for future in as_completed(futures):
                rows, seed_idx, game_id = future.result()
                if rows:
                    csv_writer.writerows(rows)
                    total_records += len(rows)
                    games_with_winner += 1
                    f.flush()

                games_done += 1
                if games_done % 20 == 0:
                    elapsed = time.time() - start_time
                    avg = elapsed / games_done
                    eta = avg * (total_games - games_done)
                    win_rate = games_with_winner / games_done * 100
                    log_message(
                        f"Progress: {games_done}/{total_games} | "
                        f"Records: {total_records} | "
                        f"Win Rate: {win_rate:.1f}% | "
                        f"Elapsed: {elapsed/3600:.2f}h | "
                        f"ETA: {eta/3600:.2f}h"
                    )

    total_time = time.time() - start_time
    log_message(f"=== DONE ===")
    log_message(f"Games: {games_done}, Winners: {games_with_winner}, Records: {total_records}")
    log_message(f"Total time: {total_time/3600:.2f}h")

if __name__ == "__main__":
    main()
