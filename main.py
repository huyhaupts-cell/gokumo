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
    """Main AlphaZero training loop"""
    
    def __init__(self, 
                 num_iterations: int = 100,
                 num_games_per_iteration: int = 32,
                 num_mcts_simulations: int = 800,
                 batch_size: int = 64,
                 epochs_per_iteration: int = 10):
        
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
        
        # Checkpoint directory
        self.checkpoint_dir = './checkpoints'
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def train(self):
        """Main training loop"""
        print(f"Training on device: {self.device}")
        print(f"Iterations: {self.num_iterations}")
        print(f"Games per iteration: {self.num_games_per_iteration}")
        print(f"MCTS simulations: {self.num_mcts_simulations}\n")
        
        for iteration in range(self.num_iterations):
            print(f"\n{'='*60}")
            print(f"Iteration {iteration + 1}/{self.num_iterations}")
            print(f"{'='*60}")
            
            # Step 1: Self-play
            print("\n[1/4] Running self-play games...")
            experiences = self._run_self_play()
            print(f"      Generated {len(experiences)} experiences")
            
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
            avg_policy_loss, avg_value_loss = self._train_network()
            print(f"      Policy loss: {avg_policy_loss:.4f}")
            print(f"      Value loss:  {avg_value_loss:.4f}")
            
            # Step 4: Save checkpoint
            print("[4/4] Saving checkpoint...")
            self._save_checkpoint(iteration)
            print(f"      Checkpoint saved")
    
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
        """Save model checkpoint"""
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f"model_iter_{iteration:04d}.pt"
        )
        torch.save({
            'iteration': iteration,
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.trainer.optimizer.state_dict(),
        }, checkpoint_path)


if __name__ == '__main__':
    # Configuration
    config = {
        'num_iterations': 100,
        'num_games_per_iteration': 32,
        'num_mcts_simulations': 800,
        'batch_size': 64,
        'epochs_per_iteration': 10
    }
    
    # Train
    agent = AlphaZeroGomoku(**config)
    agent.train()