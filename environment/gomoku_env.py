import gym
from gym import spaces
import numpy as np
from typing import Tuple, Dict

class GomokuEnv(gym.Env):
    """
    Gomoku (5 in a row) environment with 15x15 board
    Gym-compatible for easy training
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, board_size: int = 15, win_condition: int = 5):
        super(GomokuEnv, self).__init__()
        self.board_size = board_size
        self.win_condition = win_condition
        
        # Action space: 225 possible positions (15x15)
        self.action_space = spaces.Discrete(self.board_size ** 2)
        
        # Observation space: 15x15 board with values 0 (empty), 1 (player), 2 (opponent)
        self.observation_space = spaces.Box(
            low=0, high=2, 
            shape=(self.board_size, self.board_size), 
            dtype=np.int32
        )
        
        self.reset()
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int32)
        self.current_player = 1  # Player 1 starts
        self.done = False
        self.move_count = 0
        return self.board.copy()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one step of environment dynamics
        
        Args:
            action: Position to place stone (0-224 for 15x15)
        
        Returns:
            observation, reward, done, info
        """
        if self.done:
            return self.board.copy(), 0, True, {}
        
        x, y = divmod(action, self.board_size)
        
        # Check valid move
        if self.board[x, y] != 0:
            return self.board.copy(), -1, False, {"invalid_move": True}
        
        # Place stone
        self.board[x, y] = self.current_player
        self.move_count += 1
        
        # Check win
        winner = self._check_winner(x, y)
        
        if winner != 0:
            reward = 1.0 if winner == self.current_player else -1.0
            self.done = True
            return self.board.copy(), reward, True, {"winner": winner}
        
        # Check draw (board full)
        if self.move_count >= self.board_size ** 2:
            self.done = True
            return self.board.copy(), 0, True, {"draw": True}
        
        # Switch player
        self.current_player = 3 - self.current_player
        
        return self.board.copy(), 0, False, {}
    
    def _check_winner(self, x: int, y: int) -> int:
        """
        Check if there's a winner after placing stone at (x, y)
        Returns: 0 (no winner), 1 (player 1), 2 (player 2)
        """
        player = self.board[x, y]
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for dx, dy in directions:
            count = 1
            
            # Check positive direction
            nx, ny = x + dx, y + dy
            while 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                if self.board[nx, ny] == player:
                    count += 1
                    nx += dx
                    ny += dy
                else:
                    break
            
            # Check negative direction
            nx, ny = x - dx, y - dy
            while 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                if self.board[nx, ny] == player:
                    count += 1
                    nx -= dx
                    ny -= dy
                else:
                    break
            
            if count >= self.win_condition:
                return player
        
        return 0
    
    def get_valid_moves(self) -> np.ndarray:
        """Get all valid moves (empty positions)"""
        return np.where(self.board.flatten() == 0)[0]
    
    def render(self, mode: str = 'human'):
        """Render the board"""
        print("\n   ", end="")
        for i in range(self.board_size):
            print(f"{i:2}", end=" ")
        print()
        
        for i in range(self.board_size):
            print(f"{i:2} ", end="")
            for j in range(self.board_size):
                if self.board[i, j] == 0:
                    print(" .", end=" ")
                elif self.board[i, j] == 1:
                    print(" X", end=" ")
                else:
                    print(" O", end=" ")
            print()
        print()
    
    def close(self):
        pass