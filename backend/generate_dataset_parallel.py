"""
Dataset generation V3: Winner-only filtering + critical move flagging.
Uses multiprocessing for parallel game simulation.
"""
import os
import random
import csv
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from engine import (
    BOARD_SIZE, check_win_condition,
    INITIAL_POSITIONS_P1, INITIAL_POSITIONS_P2, create_initial_board
)
from ai import get_all_possible_moves, simulate_move, minimax


def run_single_game(game_id, depth=2, epsilon=0.15):
    """
    Runs a Minimax self-play game and returns ONLY the winner's moves,
    with a 'critical' flag for the last N moves before victory.
    """
    board = create_initial_board()
    current_player = 1
    history = []  # all moves
    turn_count = 0
    max_turns = 100
    visited_states = []

    while turn_count < max_turns:
        opponent = 2 if current_player == 1 else 1
        possible_moves = get_all_possible_moves(board, current_player)

        # Prevent repetition loops
        valid_moves = [
            m for m in possible_moves
            if str(simulate_move(board, m)) not in visited_states[-6:]
        ]
        if not valid_moves:
            valid_moves = possible_moves
        if not valid_moves:
            break

        # Epsilon-greedy: occasionally pick a random move for diversity
        if random.random() < epsilon:
            best_move = random.choice(valid_moves)
        else:
            best_score = -float('inf')
            best_move = None
            for move in valid_moves:
                next_board = simulate_move(board, move)
                if check_win_condition(move["targetX"], move["targetY"]):
                    best_move = move
                    break
                score = minimax(
                    next_board, depth - 1,
                    -float('inf'), float('inf'),
                    False, current_player, opponent
                )
                if score > best_score:
                    best_score = score
                    best_move = move
            if not best_move:
                best_move = valid_moves[0]

        # Record the move
        history.append({
            "board": [row[:] for row in board],
            "player": current_player,
            "from_c": best_move["pieceX"],
            "from_r": best_move["pieceY"],
            "to_c": best_move["targetX"],
            "to_r": best_move["targetY"],
        })

        # Apply move
        board = simulate_move(board, best_move)
        visited_states.append(str(board))

        if check_win_condition(best_move["targetX"], best_move["targetY"]):
            return history, current_player

        current_player = opponent
        turn_count += 1

    return history, 0  # draw


def worker(game_id):
    """Process one game and return CSV rows for WINNER's moves only."""
    # Vary depth for diversity
    r = random.random()
    if r < 0.15:
        depth = 3
    elif r < 0.55:
        depth = 2
    else:
        depth = 1
    epsilon = random.uniform(0.05, 0.25)

    try:
        history, winner = run_single_game(game_id, depth=depth, epsilon=epsilon)
        if winner == 0:
            return []

        # Filter: keep only the WINNER's moves
        winner_moves = [h for h in history if h["player"] == winner]
        total = len(winner_moves)

        results = []
        for i, step in enumerate(winner_moves):
            # Mark last 5 winner moves as 'critical'
            is_critical = 1 if (total - i) <= 5 else 0

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
            results.append(row)
        return results

    except Exception as e:
        print(f"Error in game {game_id}: {e}", flush=True)
        return []


def generate_parallel(num_games=2000, output_file="dataset_v3.csv", max_workers=32):
    print(f"Starting V3 dataset generation: {num_games} games, {max_workers} workers...", flush=True)
    print("Mode: Winner-only + Critical flag", flush=True)
    start_time = time.time()

    header = [f"cell_{y}_{x}" for y in range(BOARD_SIZE) for x in range(BOARD_SIZE)]
    header.extend([
        "player_turn",
        "action_from_c", "action_from_r",
        "action_to_c", "action_to_r",
        "winner", "is_critical"
    ])

    total_records = 0
    games_completed = 0

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        f.flush()

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_game = {executor.submit(worker, i): i for i in range(num_games)}

            for future in as_completed(future_to_game):
                result = future.result()
                if result:
                    writer.writerows(result)
                    total_records += len(result)

                games_completed += 1
                if games_completed % 50 == 0:
                    elapsed = time.time() - start_time
                    avg = elapsed / games_completed
                    eta = avg * (num_games - games_completed)
                    print(
                        f"Progress: {games_completed}/{num_games} | "
                        f"Records: {total_records} | "
                        f"Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s",
                        flush=True
                    )
                    f.flush()

    end_time = time.time()
    print(f"\nDone! {total_records} winner-only records in {end_time - start_time:.1f}s", flush=True)
    print(f"Saved to {output_file}", flush=True)


if __name__ == "__main__":
    generate_parallel(num_games=3000, max_workers=32)
