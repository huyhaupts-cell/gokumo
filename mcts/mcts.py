import numpy as np
import math
from typing import Dict, Tuple, List

class MCTSNode:
    """Node in MCTS tree"""
    def __init__(self, board_state: np.ndarray, parent=None, action: int = None):
        self.board = board_state.copy()
        self.parent = parent
        self.action = action
        self.children: Dict[int, MCTSNode] = {}
        self.visits = 0
        self.value = 0.0
        self.policy_prior = 0.0
    
    def ucb_score(self, c_puct: float = 1.25) -> float:
        """Upper Confidence Bound score"""
        if self.visits == 0:
            return float('inf')
        
        exploitation = self.value / self.visits
        exploration = c_puct * self.policy_prior * math.sqrt(self.parent.visits) / (1 + self.visits)
        return exploitation + exploration
    
    def select_child(self) -> 'MCTSNode':
        """Select child with highest UCB score"""
        return max(self.children.values(), key=lambda c: c.ucb_score())
    
    def expand(self, valid_moves: np.ndarray, policy_priors: np.ndarray):
        """Expand node with children"""
        for move in valid_moves:
            if move not in self.children:
                self.children[move] = MCTSNode(self.board, parent=self, action=move)
                self.children[move].policy_prior = policy_priors[move]
    
    def backup(self, value: float):
        """Backup value through tree"""
        self.visits += 1
        self.value += value
        
        if self.parent is not None:
            self.parent.backup(-value)  # Negate for alternating player

class MCTS:
    """Monte Carlo Tree Search with neural network guidance"""
    
    def __init__(self, env, network, num_simulations: int = 800, c_puct: float = 1.25):
        self.env = env
        self.network = network
        self.num_simulations = num_simulations
        self.c_puct = c_puct
    
    def search(self, board: np.ndarray, player: int) -> Tuple[np.ndarray, float]:
        """
        Run MCTS search
        
        Returns:
            action_probs: (board_size^2,) probability distribution over actions
            value: estimated value of position
        """
        root = MCTSNode(board)
        
        for _ in range(self.num_simulations):
            node = root
            
            # Selection & Expansion
            while node.visits > 0 and len(node.children) > 0:
                node = node.select_child()
            
            if node.visits > 0:
                # Expand
                valid_moves = np.where(node.board.flatten() == 0)[0]
                if len(valid_moves) > 0:
                    # Get policy and value from network
                    input_tensor = self.network.prepare_input(node.board, player)
                    with torch.no_grad():
                        policy, value = self.network(input_tensor)
                    
                    policy_priors = torch.exp(policy[0]).detach().numpy()
                    value = value[0, 0].item()
                    
                    node.expand(valid_moves, policy_priors)
                    
                    # Select first child for simulation
                    if node.children:
                        node = list(node.children.values())[0]
            else:
                # Evaluate with network
                input_tensor = self.network.prepare_input(node.board, player)
                with torch.no_grad():
                    policy, value = self.network(input_tensor)
                value = value[0, 0].item()
            
            # Backup
            node.backup(value)
        
        # Return visit counts as action probabilities
        action_probs = np.zeros(self.env.board_size ** 2)
        for move, child in root.children.items():
            action_probs[move] = child.visits
        
        action_probs = action_probs / action_probs.sum()
        value = root.value / max(1, root.visits)
        
        return action_probs, value


import torch