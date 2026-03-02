import torch
import numpy as np
from environment.gomoku_env import GomokuEnv
from network.network import GomokuNet
from mcts.mcts import MCTS
from replay_buffer.buffer import ReplayBuffer
from self_play.self_play import SelfPlayParallel, SelfPlayGame
from training.trainer import Trainer
import os
from datetime import datetime

class AlphaZeroGomoku:
    """Main AlphaZero training loop with resume capability"""
    
    def __init__(self, 
                 num_iterations: int = 100,
                 num_games_per_iteration: int = 32,
                 num_mcts_simulations: int = 800,
                 batch_size: int = 64,
                 epochs_per_iteration: int = 10,
                 resume_from_checkpoint: str = None):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self.env = GomokuEnv(board_size=15, win_condition=5)
        self.network = GomokuNet(board_size=15, num_residual_blocks=10, channels=128).to(self.device)
        self.trainer = Trainer(self.network, lr=0.001, l2_regularization=1e-4)
        self.replay_buffer = ReplayBuffer(capacity=100000)
        self.self_play = SelfPlayParallel(num_workers=4, num_games=num_games_per_iteration)
        
        # Hyperparameters
        self.num_iterations = num_iterations
        self.num_games_per_iteration = num_games_per_iteration
        self.num_mcts_simulations = num_mcts_simulations
        self.batch_size = batch_size
        self.epochs_per_iteration = epochs_per_iteration
        
        # Training state
        self.start_iteration = 0
        self.training_history = {
            'iterations': [],
            'policy_losses': [],
            'value_losses': [],
            'times': []
        }
        
        # Checkpoint directory
        self.checkpoint_dir = './checkpoints'
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # ✅ NEW: Resume from checkpoint
        if resume_from_checkpoint:
            self.load_checkpoint(resume_from_checkpoint)
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load training state from checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"\n{'='*60}")
        print(f"Loading checkpoint: {checkpoint_path}")
        print(f"{'='*60}\n")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model
        self.network.load_state_dict(checkpoint['model_state_dict'])
        print("✓ Model weights loaded")
        
        # Load optimizer state
        self.trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("✓ Optimizer state loaded")
        
        # Load training state
        if 'iteration' in checkpoint:
            self.start_iteration = checkpoint['iteration'] + 1
            print(f"✓ Resuming from iteration {self.start_iteration}")
        
        if 'training_history' in checkpoint:
            self.training_history = checkpoint['training_history']
            print(f"✓ Training history loaded ({len(self.training_history['iterations'])} iterations)")
        
        print()
    
    def train(self):
        """Main training loop (with resume support)"""
        print(f"Training on device: {self.device}")
        print(f"Iterations: {self.start_iteration} → {self.num_iterations}")
        print(f"Games per iteration: {self.num_games_per_iteration}")
        print(f"MCTS simulations: {self.num_mcts_simulations}\n")
        
        for iteration in range(self.start_iteration, self.num_iterations):
            print(f"\n{'='*60}")
            print(f"Iteration {iteration + 1}/{self.num_iterations}")
            print(f"{'='*60}")
            
            import time
            iter_start = time.time()
            
            # Step 1: Self-play
            print("\n[1/4] Running self-play games...")
            sp_start = time.time()
            experiences = self._run_self_play()
            sp_time = time.time() - sp_start
            print(f"      Generated {len(experiences)} experiences ({sp_time:.1f}s)")
            
            # Step 2: Add to replay buffer
            print("[2/4] Adding experiences to replay buffer...")
            for exp in experiences:
                self.replay_buffer.add(
                    exp['board'],
                    exp['action_probs'],
                    exp['value'],
                    exp['outcome']
                )
            print(f"      Replay buffer size: {len(self.replay_buffer)}")
            
            # Step 3: Train network
            print(f"[3/4] Training network ({self.epochs_per_iteration} epochs)...")
            tr_start = time.time()
            avg_policy_loss, avg_value_loss = self._train_network()
            tr_time = time.time() - tr_start
            print(f"      Policy loss: {avg_policy_loss:.4f}")
            print(f"      Value loss:  {avg_value_loss:.4f}")
            print(f"      Time: {tr_time:.1f}s")
            
            # Step 4: Save checkpoint
            print("[4/4] Saving checkpoint...")
            self._save_checkpoint(iteration)
            print(f"      Checkpoint saved")
            
            # Track history
            iter_time = time.time() - iter_start
            self.training_history['iterations'].append(iteration)
            self.training_history['policy_losses'].append(avg_policy_loss)
            self.training_history['value_losses'].append(avg_value_loss)
            self.training_history['times'].append(iter_time)
            
            # ETA
            avg_iter_time = np.mean(self.training_history['times'][-10:])
            remaining_iters = self.num_iterations - (iteration + 1)
            eta_seconds = avg_iter_time * remaining_iters
            eta_hours = eta_seconds / 3600
            
            print(f"\nIteration time: {iter_time:.1f}s | ETA: {eta_hours:.1f}h")
    
    def _run_self_play(self) -> list:
        """Run self-play games and collect experiences"""
        all_experiences = []
        
        for game_id in range(self.num_games_per_iteration):
            env = GomokuEnv()
            mcts = MCTS(env, self.network, num_simulations=self.num_mcts_simulations)
            game = SelfPlayGame(env, self.network, mcts, temperature=1.0)
            
            try:
                experiences = game.play()
                all_experiences.extend(experiences)
            except Exception as e:
                print(f"      Error in game {game_id}: {e}")
                continue
        
        return all_experiences
    
    def _train_network(self) -> tuple:
        """Train network on replay buffer"""
        if len(self.replay_buffer) == 0:
            return 0.0, 0.0
        
        return self.trainer.train_epoch(
            self.replay_buffer,
            batch_size=self.batch_size,
            epochs=self.epochs_per_iteration,
            device=self.device
        )
    
    def _save_checkpoint(self, iteration: int):
        """Save model checkpoint with full training state"""
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f"model_iter_{iteration:04d}.pt"
        )
        torch.save({
            'iteration': iteration,
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.trainer.optimizer.state_dict(),
            'config': {
                'num_iterations': self.num_iterations,
                'num_games_per_iteration': self.num_games_per_iteration,
                'num_mcts_simulations': self.num_mcts_simulations,
                'batch_size': self.batch_size,
                'epochs_per_iteration': self.epochs_per_iteration,
            },
            'training_history': self.training_history,
        }, checkpoint_path)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train AlphaZero Gomoku')
    parser.add_argument('--num-iterations', type=int, default=100,
                       help='Total number of iterations')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint (e.g., checkpoints/model_iter_0030.pt)')
    parser.add_argument('--auto-resume', action='store_true',
                       help='Auto-resume from latest checkpoint')
    parser.add_argument('--games-per-iter', type=int, default=32,
                       help='Number of games per iteration')
    parser.add_argument('--mcts-sims', type=int, default=800,
                       help='Number of MCTS simulations per move')
    
    args = parser.parse_args()
    
    # Auto-find latest checkpoint
    resume_path = args.resume
    if args.auto_resume and resume_path is None:
        checkpoint_dir = './checkpoints'
        if os.path.exists(checkpoint_dir):
            model_files = list(os.listdir(checkpoint_dir))
            model_files = [f for f in model_files if f.startswith('model_iter_') and f.endswith('.pt')]
            if model_files:
                model_files.sort(key=lambda x: int(x.split('_')[-1].replace('.pt', '')))
                resume_path = os.path.join(checkpoint_dir, model_files[-1])
                print(f"Auto-found latest checkpoint: {resume_path}\n")
    
    # Configuration
    config = {
        'num_iterations': args.num_iterations,
        'num_games_per_iteration': args.games_per_iter,
        'num_mcts_simulations': args.mcts_sims,
        'batch_size': 64,
        'epochs_per_iteration': 10,
        'resume_from_checkpoint': resume_path,
    }
    
    # Train
    agent = AlphaZeroGomoku(**config)
    agent.train()