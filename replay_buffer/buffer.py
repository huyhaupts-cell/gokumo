import numpy as np
from collections import deque
import random
from typing import Tuple

class ReplayBuffer:
    """Replay buffer for storing self-play experience"""
    
    def __init__(self, capacity: int = 100000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def add(self, board: np.ndarray, action_probs: np.ndarray, 
            value: float, game_outcome: float):
        """
        Add experience to buffer
        
        Args:
            board: Board state (board_size, board_size)
            action_probs: MCTS action probabilities (board_size^2,)
            value: Network value estimate
            game_outcome: Final game outcome (+1, 0, -1)
        """
        self.buffer.append({
            'board': board.copy(),
            'action_probs': action_probs.copy(),
            'value': value,
            'outcome': game_outcome
        })
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample batch from buffer
        
        Returns:
            boards: (batch_size, board_size, board_size)
            action_probs: (batch_size, board_size^2)
            outcomes: (batch_size,)
        """
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        
        boards = np.array([b['board'] for b in batch])
        action_probs = np.array([b['action_probs'] for b in batch])
        outcomes = np.array([b['outcome'] for b in batch])
        
        return boards, action_probs, outcomes
    
    def __len__(self) -> int:
        return len(self.buffer)