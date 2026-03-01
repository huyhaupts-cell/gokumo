import torch
import numpy as np
import sys
from pathlib import Path

# Add project to path
PROJECT_PATH = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_PATH))

from environment.gomoku_env import GomokuEnv
from network.network import GomokuNet
from mcts.mcts import MCTS

class HumanVsAI:
    """Interactive game: Human vs AI"""
    
    def __init__(self, model_path: str, num_mcts_simulations: int = 400):
        """
        Args:
            model_path: Path to trained model checkpoint
            num_mcts_simulations: Number of MCTS simulations per move
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.network = GomokuNet(board_size=15, num_residual_blocks=10, channels=128).to(self.device)
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.network.eval()
        print("✓ Model loaded!\n")
        
        # Environment
        self.env = GomokuEnv(board_size=15, win_condition=5)
        
        # MCTS
        self.mcts = MCTS(self.env, self.network, num_simulations=num_mcts_simulations)
        self.num_mcts_simulations = num_mcts_simulations
    
    def play(self, ai_player: int = 2, human_first: bool = False):
        """
        Play interactive game
        
        Args:
            ai_player: 1 or 2 (which player is AI)
            human_first: If True, human moves first
        """
        self.env.reset()
        current_player = 1 if human_first else 2
        
        print("="*60)
        print("GOMOKU: Human vs AI")
        print("="*60)
        print(f"\nAI is Player {ai_player} ({'X' if ai_player == 1 else 'O'})")
        print(f"You are Player {3 - ai_player} ({'O' if ai_player == 1 else 'X'})")
        print(f"First move: {'You' if human_first else 'AI'}\n")
        
        move_count = 0
        
        while not self.env.done:
            self.env.render()
            
            if current_player == ai_player:
                # AI move
                print(f"[Move {move_count + 1}] AI thinking...", end=" ", flush=True)
                action, confidence = self._get_ai_move()
                x, y = divmod(action, 15)
                print(f"AI places at ({x}, {y}) | Confidence: {confidence:.2%}\n")
                
                self.env.step(action)
            else:
                # Human move
                while True:
                    try:
                        human_input = input(f"[Move {move_count + 1}] Your move (x,y) or 'help': ").strip()
                        
                        if human_input.lower() == 'help':
                            self._show_suggestions()
                            continue
                        
                        x, y = map(int, human_input.split(','))
                        
                        if not (0 <= x < 15 and 0 <= y < 15):
                            print("❌ Out of bounds! (0-14)\n")
                            continue
                        
                        action = x * 15 + y
                        
                        if self.env.board[x, y] != 0:
                            print("❌ Position already occupied!\n")
                            continue
                        
                        self.env.step(action)
                        print()
                        break
                    
                    except ValueError:
                        print("❌ Invalid input! Use format: x,y (e.g., 7,7)\n")
                        continue
            
            # Switch player
            current_player = 3 - current_player
            move_count += 1
        
        # Game end
        self._show_result(move_count)
    
    def _get_ai_move(self) -> tuple:
        """
        Get AI move using MCTS
        
        Returns:
            action: Best move
            confidence: Probability of best move
        """
        board = self.env.board.copy()
        player = self.env.current_player
        
        with torch.no_grad():
            action_probs, value = self.mcts.search(board, player)
        
        # Select best move
        best_action = np.argmax(action_probs)
        confidence = action_probs[best_action]
        
        return best_action, confidence
    
    def _show_suggestions(self):
        """Show AI suggestions for all valid moves"""
        board = self.env.board.copy()
        player = self.env.current_player
        
        print("\nAI Suggestions:")
        print("-" * 40)
        
        with torch.no_grad():
            action_probs, _ = self.mcts.search(board, player)
        
        # Get top 5 moves
        top_moves = np.argsort(action_probs)[-5:][::-1]
        
        for i, action in enumerate(top_moves, 1):
            x, y = divmod(action, 15)
            prob = action_probs[action]
            if prob > 0:
                print(f"{i}. ({x:2d}, {y:2d}) - {prob:6.2%}")
        
        print()
    
    def _show_result(self, move_count: int):
        """Show game result"""
        print("\n" + "="*60)
        print("GAME OVER!")
        print("="*60)
        
        # Determine winner
        last_move = np.where(self.env.board != 0)
        if len(last_move[0]) > 0:
            x, y = last_move[0][-1], last_move[1][-1]
            winner = self.env.board[x, y]
            
            if winner == 1:
                print("🎉 Player 1 (X) WINS!")
            else:
                print("🎉 Player 2 (O) WINS!")
        else:
            print("Draw!")
        
        print(f"Total moves: {move_count}")
        print("="*60)
    
    def play_multiple_games(self, num_games: int = 5):
        """Play multiple games, track statistics"""
        results = {'ai_wins': 0, 'human_wins': 0, 'draws': 0}
        
        for game_id in range(num_games):
            print(f"\n\n{'='*60}")
            print(f"Game {game_id + 1}/{num_games}")
            print(f"{'='*60}\n")
            
            self.play(ai_player=2)
            
            # Track result (simple version)
            # In real implementation, properly track winner
        
        print(f"\nStatistics:")
        print(f"AI Wins: {results['ai_wins']}")
        print(f"Human Wins: {results['human_wins']}")
        print(f"Draws: {results['draws']}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Play Gomoku vs AI')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--simulations', type=int, default=400,
                       help='MCTS simulations per move')
    parser.add_argument('--ai-player', type=int, default=2,
                       help='AI player number (1 or 2)')
    parser.add_argument('--human-first', action='store_true',
                       help='Human moves first')
    
    args = parser.parse_args()
    
    # Create game
    game = HumanVsAI(args.model, num_mcts_simulations=args.simulations)
    
    # Play
    game.play(ai_player=args.ai_player, human_first=args.human_first)


if __name__ == '__main__':
    main()