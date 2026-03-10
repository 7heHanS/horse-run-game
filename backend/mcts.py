"""
Monte Carlo Tree Search (MCTS) implementation for AI Policy Network V4.
Integrates PUCT selection, NN-based expansion w/ caching, and heuristic evaluation w/ tanh normalization.
"""
import math
import torch
import numpy as np

from engine import BOARD_SIZE, check_win_condition
from ai import get_all_possible_moves, simulate_move, evaluate_board

try:
    from model_v4 import board_to_channels, get_v4_masks
except ImportError:
    pass

def hash_board(board, player):
    """Create a hashable representation of the board state for caching."""
    return tuple(tuple(row) for row in board), player

class MCTSNode:
    def __init__(self, state, player, prior_prob, parent=None, move=None):
        self.state = state
        self.player = player
        self.prior_prob = prior_prob
        self.parent = parent
        self.move = move
        
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0

    @property
    def q_value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def expand(self, action_probs):
        """action_probs is a list of tuples: (move_dict, probability)"""
        for move, prob in action_probs:
            next_state = simulate_move(self.state, move)
            next_player = 2 if self.player == 1 else 1
            
            # Unique key for the move based on coordinates
            move_key = (move["pieceX"], move["pieceY"], move["targetX"], move["targetY"])
            
            if move_key not in self.children:
                self.children[move_key] = MCTSNode(
                    state=next_state,
                    player=next_player,
                    prior_prob=prob,
                    parent=self,
                    move=move
                )

    def is_expanded(self):
        return len(self.children) > 0

class MCTS:
    def __init__(self, model, device, c_puct=2.5, num_simulations=100):
        self.model = model
        self.device = device
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.nn_cache = {}  # Inference speedup: caches (board, player) -> action_probs

    def get_action_probs(self, board, player):
        state_key = hash_board(board, player)
        if state_key in self.nn_cache:
            return self.nn_cache[state_key]

        possible_moves = get_all_possible_moves(board, player)
        if not possible_moves:
            return []

        channels = board_to_channels(board, player).unsqueeze(0).to(self.device)
        from_mask, _ = get_v4_masks(board, player)
        from_mask = from_mask.to(self.device)

        with torch.no_grad():
            logits_from, logits_to = self.model(channels)
            logits_from[0][~from_mask] = -1e9
            
            probs_from = torch.nn.functional.softmax(logits_from[0], dim=0)
            probs_to = torch.nn.functional.softmax(logits_to[0], dim=0)

        action_probs = []
        for move in possible_moves:
            from_idx = move["pieceY"] * BOARD_SIZE + move["pieceX"]
            to_idx = move["targetY"] * BOARD_SIZE + move["targetX"]
            prob = probs_from[from_idx].item() * probs_to[to_idx].item()
            action_probs.append((move, prob))
            
        # Normalize sum to 1.0
        total_prob = sum(p for _, p in action_probs)
        if total_prob > 0:
            action_probs = [(m, p / total_prob) for m, p in action_probs]

        self.nn_cache[state_key] = action_probs
        return action_probs

    def evaluate(self, board, eval_player):
        """
        Uses existing AI heuristic with tanh normalization to [-1, 1].
        """
        opponent = 2 if eval_player == 1 else 1
        raw_score = evaluate_board(board, ai_player=eval_player, human_player=opponent)
        
        # Scaling factor to squash huge heuristic values (WIN=100000, MEADOW=500) into reasonable margins
        return math.tanh(raw_score / 2000.0)

    def search(self, initial_board, player, add_noise=False):
        root = MCTSNode(state=initial_board, player=player, prior_prob=1.0)
        
        # Expand root
        action_probs = self.get_action_probs(initial_board, player)
        root.expand(action_probs)

        if not root.is_expanded():
            return None

        if add_noise and len(root.children) > 0:
            # Dirichlet noise on root to encourage exploration (AlphaGo technique)
            dirichlet_alpha = 0.3
            noise = np.random.dirichlet([dirichlet_alpha] * len(root.children))
            for i, child in enumerate(root.children.values()):
                child.prior_prob = 0.75 * child.prior_prob + 0.25 * noise[i]

        for sim in range(self.num_simulations):
            node = root
            search_path = [node]

            # 1. Selection
            while node.is_expanded():
                best_score = -float('inf')
                best_child = None
                
                for child in node.children.values():
                    # PUCT Formula: Q + c * P * sqrt(N_parent) / (1 + N_child)
                    # We negate child.q_value because Q is from the child's perspective (opponent)
                    q = -child.q_value 
                    u = self.c_puct * child.prior_prob * math.sqrt(node.visit_count) / (1 + child.visit_count)
                    score = q + u

                    if score > best_score:
                        best_score = score
                        best_child = child
                        
                node = best_child
                search_path.append(node)

            # 2. Evaluation / Expansion
            is_terminal = False
            value = 0.0
            
            # Did the last move win the game?
            if node.move and check_win_condition(node.move["targetX"], node.move["targetY"]):
                is_terminal = True
                # The parent player made the winning move. So for the current node's player, they lost (-1.0).
                value = -1.0 
            
            if not is_terminal:
                action_probs = self.get_action_probs(node.state, node.player)
                if not action_probs:
                    value = -1.0 # Terminal loss due to no moves
                else:
                    node.expand(action_probs)
                    value = self.evaluate(node.state, node.player)
            
            # 3. Backpropagation
            # The value flips at each step because it's a zero-sum turn-based game
            for n in reversed(search_path):
                n.value_sum += value
                n.visit_count += 1
                value = -value

        # Return the move that was visited the most (most robust search branch)
        best_child = max(root.children.values(), key=lambda c: c.visit_count)
        return best_child.move
