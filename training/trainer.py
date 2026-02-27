import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple

class Trainer:
    """Network trainer"""
    
    def __init__(self, network, lr: float = 0.001, l2_regularization: float = 1e-4):
        self.network = network
        self.optimizer = optim.Adam(network.parameters(), lr=lr, weight_decay=l2_regularization)
        self.policy_loss_fn = nn.KLDivLoss()
        self.value_loss_fn = nn.MSELoss()
    
    def train_step(self, boards: np.ndarray, action_probs: np.ndarray, 
                   outcomes: np.ndarray, device: str = 'cpu') -> Tuple[float, float]:
        """
        Single training step
        
        Returns:
            policy_loss, value_loss
        """
        # Convert to tensors
        boards_tensor = torch.FloatTensor(boards).to(device)
        action_probs_tensor = torch.FloatTensor(action_probs).to(device)
        outcomes_tensor = torch.FloatTensor(outcomes).to(device)
        
        # Prepare input
        batch_size = boards.shape[0]
        input_tensor = torch.zeros((batch_size, 3, boards.shape[1], boards.shape[2]))
        
        for i in range(batch_size):
            board = boards[i]
            player = 1  # Assume player 1 (can be fixed or varied)
            input_tensor[i, 0] = torch.FloatTensor(board == player)
            input_tensor[i, 1] = torch.FloatTensor(board == (3 - player))
            input_tensor[i, 2] = player
        
        input_tensor = input_tensor.to(device)
        
        # Forward pass
        policy_logits, value_pred = self.network(input_tensor)
        
        # Policy loss
        policy_loss = self.policy_loss_fn(policy_logits, action_probs_tensor)
        
        # Value loss
        value_loss = self.value_loss_fn(value_pred.squeeze(), outcomes_tensor)
        
        # Combined loss
        total_loss = policy_loss + value_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return policy_loss.item(), value_loss.item()
    
    def train_epoch(self, replay_buffer, batch_size: int = 64, 
                   epochs: int = 10, device: str = 'cpu') -> Tuple[float, float]:
        """
        Train for one epoch
        
        Returns:
            Average policy loss, average value loss
        """
        total_policy_loss = 0.0
        total_value_loss = 0.0
        num_batches = 0
        
        for _ in range(epochs):
            boards, action_probs, outcomes = replay_buffer.sample(batch_size)
            
            if len(boards) > 0:
                p_loss, v_loss = self.train_step(boards, action_probs, outcomes, device)
                total_policy_loss += p_loss
                total_value_loss += v_loss
                num_batches += 1
        
        avg_policy_loss = total_policy_loss / max(1, num_batches)
        avg_value_loss = total_value_loss / max(1, num_batches)
        
        return avg_policy_loss, avg_value_loss