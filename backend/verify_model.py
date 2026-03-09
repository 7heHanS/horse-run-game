"""
Verification script V3: Benchmarks the DL model against Minimax.
Loads the model ONCE for fast inference. Supports Legal Move Masking.
"""
import torch
import sys
import os

from engine import BOARD_SIZE, check_win_condition, create_initial_board
from ai import get_all_possible_moves, simulate_move, minimax

try:
    from model_v4 import HorseRunPolicyNetV4 as PolicyNetV4, board_to_channels, get_v4_masks, NUM_POSITIONS
    MODEL_CLASS = "V4"
except ImportError:
    try:
        from model_v3 import HorseRunPolicyNetV3, board_to_channels, get_legal_move_mask, NUM_POSITIONS
        MODEL_CLASS = "V3"
    except ImportError:
        from train_model_v2 import AIPolicyNetV2
        MODEL_CLASS = "V2"


def get_ml_move_fast(model, board, current_player, device, model_class):
    """Get ML move using pre-loaded model with legal move masking."""
    possible_moves = get_all_possible_moves(board, current_player)
    if not possible_moves:
        return None

    if model_class == "V4+MCTS":
        return model.search(board, current_player)

    elif model_class == "V4":
        channels = board_to_channels(board, current_player).unsqueeze(0).to(device)
        from_mask, _ = get_v4_masks(board, current_player)
        from_mask = from_mask.to(device)

        with torch.no_grad():
            logits_from, logits_to = model(channels)
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
                
    elif model_class == "V3":
        channels = board_to_channels(board, current_player).unsqueeze(0).to(device)
        legal_mask = get_legal_move_mask(board, current_player).to(device)

        with torch.no_grad():
            logits = model(channels)
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
    else:
        flat_board = [c for row in board for c in row]
        inp = torch.tensor(flat_board + [current_player], dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(inp)
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


def run_match(ml_model_path, minimax_depth=4, num_games=10, use_mcts=False, mcts_simulations=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if MODEL_CLASS == "V4" and "v4" in ml_model_path:
        nn_model = PolicyNetV4().to(device)
        nn_model.load_state_dict(torch.load(ml_model_path, map_location=device, weights_only=True))
        nn_model.eval()
        if use_mcts:
            from mcts import MCTS
            model = MCTS(nn_model, device, num_simulations=mcts_simulations)
            mc = "V4+MCTS"
        else:
            model = nn_model
            mc = "V4"
    elif MODEL_CLASS == "V3" and "v3" in ml_model_path:
        model = HorseRunPolicyNetV3().to(device)
        model.load_state_dict(torch.load(ml_model_path, map_location=device, weights_only=True))
        model.eval()
        mc = "V3"
    else:
        model = AIPolicyNetV2().to(device)
        model.load_state_dict(torch.load(ml_model_path, map_location=device, weights_only=True))
        model.eval()
        mc = "V2"

    win_count = 0
    draw_count = 0
    loss_count = 0

    print(f"=== {mc} Model (Masked) vs Minimax Depth {minimax_depth} ({num_games} games) ===", flush=True)

    for i in range(num_games):
        board = create_initial_board()
        ml_player = 1 if i % 2 == 0 else 2
        minimax_player = 2 if ml_player == 1 else 1
        current_player = 1
        turn_count = 0
        max_turns = 120

        print(f"Game {i+1}/{num_games}: ML=P{ml_player}", end=" ", flush=True)

        while turn_count < max_turns:
            if current_player == ml_player:
                move = get_ml_move_fast(model, board, current_player, device, mc)
                if not move:
                    possible = get_all_possible_moves(board, current_player)
                    if not possible: break
                    move = possible[0]
            else:
                possible = get_all_possible_moves(board, current_player)
                if not possible: break
                best_score = -float('inf')
                move = None
                for m in possible:
                    nb = simulate_move(board, m)
                    if check_win_condition(m["targetX"], m["targetY"]):
                        move = m
                        break
                    score = minimax(nb, minimax_depth - 1, -float('inf'), float('inf'), False, minimax_player, ml_player)
                    if score > best_score:
                        best_score = score
                        move = m
                if not move: move = possible[0]

            board = simulate_move(board, move)
            if check_win_condition(move["targetX"], move["targetY"]):
                if current_player == ml_player:
                    print(f"-> ML Won (turn {turn_count})", flush=True)
                    win_count += 1
                else:
                    print(f"-> Minimax Won (turn {turn_count})", flush=True)
                    loss_count += 1
                break
            current_player = 1 if current_player == 2 else 2
            turn_count += 1
        else:
            print("-> Draw", flush=True)
            draw_count += 1

    print(f"\n{'='*50}", flush=True)
    print(f"ML Wins:      {win_count}", flush=True)
    print(f"Minimax Wins: {loss_count}", flush=True)
    print(f"Draws:        {draw_count}", flush=True)
    print(f"Win Rate:     {(win_count / num_games) * 100:.1f}%", flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    
    use_mcts = "--mcts" in args
    if use_mcts:
        args.remove("--mcts")
        
    mcts_sims = 100
    if "--sims" in args:
        idx = args.index("--sims")
        mcts_sims = int(args[idx+1])
        args.pop(idx+1)
        args.pop(idx)

    model_path = args[0] if len(args) > 0 else "model_v4.pt"
    depth = int(args[1]) if len(args) > 1 else 4
    num_games = int(args[2]) if len(args) > 2 else 10
    
    run_match(model_path, depth, num_games, use_mcts=use_mcts, mcts_simulations=mcts_sims)
