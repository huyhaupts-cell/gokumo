"""
Evaluate model by playing against random player
"""

import torch
import numpy as np
import sys
from pathlib import Path
from tqdm import tqdm

PROJECT_PATH = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_PATH))

from environment.gomoku_env import GomokuEnv
from network.network import GomokuNet
from mcts.mcts import MCTS

class ModelEvaluator:
    """Evaluate model performance"""
    
    def __init__(self, model_path: str, num_mcts_simulations: int = 400):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.network = GomokuNet(board_size=15, num_residual_blocks=6, channels=64).to(self.device)
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.network.eval()
        
        self.num_mcts_simulations = num_mcts_simulations
    
    def play_vs_random(self, num_games: int = 10) -> dict:
        """
        Play against random player
        
        Returns:
            stats: Win rate, avg game length, etc.
        """
        results = {
            'ai_wins': 0,
            'random_wins': 0,
            'draws': 0,
            'game_lengths': [],
            'ai_as_first': 0,
            'ai_as_second': 0,
        }
        
        for game_id in tqdm(range(num_games), desc="Evaluating"):
            env = GomokuEnv()
            mcts = MCTS(env, self.network, num_simulations=self.num_mcts_simulations)
            
            ai_player = 1 if game_id % 2 == 0 else 2
            
            if ai_player == 1:
                results['ai_as_first'] += 1
            else:
                results['ai_as_second'] += 1
            
            winner = self._play_game(env, mcts, ai_player)
            
            if winner == ai_player:
                results['ai_wins'] += 1
            elif winner == 0:
                results['draws'] += 1
            else:
                results['random_wins'] += 1
            
            results['game_lengths'].append(len(np.where(env.board != 0)[0]))
        
        return results
    
    def _play_game(self, env: GomokuEnv, mcts: MCTS, ai_player: int) -> int:
        """Play single game, return winner"""
        env.reset()
        
        while not env.done:
            if env.current_player == ai_player:
                # AI move
                action_probs, _ = mcts.search(env.board, ai_player)
                action = np.argmax(action_probs)
            else:
                # Random move
                valid_moves = np.where(env.board.flatten() == 0)[0]
                action = np.random.choice(valid_moves)
            
            env.step(action)
        
        # Determine winner (simple version)
        return 1  # Placeholder
    
    def print_results(self, results: dict):
        """Print evaluation results"""
        total_games = results['ai_wins'] + results['random_wins'] + results['draws']
        win_rate = results['ai_wins'] / total_games * 100
        
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"Total games: {total_games}")
        print(f"AI wins: {results['ai_wins']}")
        print(f"Random wins: {results['random_wins']}")
        print(f"Draws: {results['draws']}")
        print(f"Win rate: {win_rate:.1f}%")
        print(f"Avg game length: {np.mean(results['game_lengths']):.0f} moves")
        print(f"AI as first: {results['ai_as_first']} games")
        print(f"AI as second: {results['ai_as_second']} games")
        print("="*60)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--num-games', type=int, default=10)
    parser.add_argument('--simulations', type=int, default=400)
    
    args = parser.parse_args()
    
    evaluator = ModelEvaluator(args.model, args.simulations)
    results = evaluator.play_vs_random(args.num_games)
    evaluator.print_results(results)