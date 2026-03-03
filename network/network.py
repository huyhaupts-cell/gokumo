import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ResidualBlock(nn.Module):
    """Residual block for deep network"""
    def __init__(self, channels: int):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

class GomokuNet(nn.Module):
    """
    AlphaZero-style network: Shared trunk + Policy head + Value head
    """
    def __init__(self, board_size: int = 15, num_residual_blocks: int = 10, channels: int = 128):
        super(GomokuNet, self).__init__()
        self.board_size = board_size
        
        # Shared trunk
        self.conv_input = nn.Conv2d(3, channels, kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(channels)
        
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_residual_blocks)]
        )
        
        # Policy head
        self.policy_conv = nn.Conv2d(channels, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, board_size * board_size)
        
        # Value head
        self.value_conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, 256)
        self.value_fc2 = nn.Linear(256, 1)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, 3, board_size, board_size)
               Channel 0: player stones
               Channel 1: opponent stones
               Channel 2: current player indicator
        
        Returns:
            policy: (batch_size, board_size * board_size)
            value: (batch_size, 1)
        """
        # Shared trunk
        trunk = F.relu(self.bn_input(self.conv_input(x)))
        trunk = self.residual_blocks(trunk)
        
        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(trunk)))
        policy = policy.view(policy.size(0), -1)
        policy = self.policy_fc(policy)
        policy = F.log_softmax(policy, dim=1)
        
        # Value head
        value = F.relu(self.value_bn(self.value_conv(trunk)))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy, value
    
    def prepare_input(self, board: np.ndarray, player: int) -> torch.Tensor:
        """
        Prepare input tensor from board state
        
        Args:
            board: (board_size, board_size) with 0, 1, 2 values
            player: current player (1 or 2)
        
        Returns:
            Input tensor (3, board_size, board_size)
        """
        board_size = board.shape[0]
        input_tensor = np.zeros((3, board_size, board_size), dtype=np.float32)
        
        # Channel 0: current player stones
        input_tensor[0] = (board == player).astype(np.float32)
        
        # Channel 1: opponent stones
        opponent = 3 - player
        input_tensor[1] = (board == opponent).astype(np.float32)
        
        # Channel 2: current player indicator
        input_tensor[2] = np.ones((board_size, board_size), dtype=np.float32) * player
        
        return torch.tensor(
                    input_tensor,
                    dtype=torch.float32,
                    device=next(self.parameters()).device
                    ).unsqueeze(0)
def prepare_batch(self, boards, players):
    batch_size = len(boards)
    board_size = boards[0].shape[0]

    input_tensor = np.zeros((batch_size, 3, board_size, board_size),
                            dtype=np.float32)

    for i in range(batch_size):
        board = boards[i]
        player = players[i]

        input_tensor[i, 0] = (board == player)
        input_tensor[i, 1] = (board == (3 - player))
        input_tensor[i, 2] = player

    return torch.tensor(
        input_tensor,
        dtype=torch.float32,
        device=next(self.parameters()).device
    )