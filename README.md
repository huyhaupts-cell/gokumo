# AlphaZero Gomoku (5 in a row)

Implementation of AlphaZero algorithm for Gomoku (15x15 board, 5 in a row to win).

## Architecture

```
Self-play (Parallel) → Replay Buffer → Train Network → Update Agent → Repeat
```

### Components

1. **Environment** (`environment/gomoku_env.py`)
   - Gym-compatible 15x15 Gomoku environment
   - Valid move checking, win detection
   - Observation: Board state (0=empty, 1=player, 2=opponent)
   - Action: Position on board (0-224)

2. **Network** (`network/network.py`)
   - AlphaZero-style dual-head architecture
   - Shared trunk: Residual blocks
   - Policy head: Outputs move probabilities
   - Value head: Outputs game outcome prediction (-1 to 1)

3. **MCTS** (`mcts/mcts.py`)
   - Monte Carlo Tree Search with neural network guidance
   - UCB (Upper Confidence Bound) for node selection
   - Policy priors from network for exploration guidance

4. **Replay Buffer** (`replay_buffer/buffer.py`)
   - Stores self-play experiences
   - Samples mini-batches for training
   - Capacity: 100,000 games

5. **Self-Play** (`self_play/self_play.py`)
   - Parallel game generation
   - Temperature-controlled action selection
   - Experience collection

6. **Trainer** (`training/trainer.py`)
   - Policy loss: KL divergence (MCTS probs vs network output)
   - Value loss: MSE (network estimate vs game outcome)
   - Gradient clipping, L2 regularization

## Training Pipeline

```
Iteration 1:
  [1] Self-play 32 games × 800 MCTS simulations each
  [2] Store 10,000+ experiences in replay buffer
  [3] Train 10 epochs on batch_size=64
  [4] Save checkpoint

Iteration 2-100: Repeat...
```

## Usage

```bash
python main.py
```

## Hyperparameters

- Board size: 15×15
- Win condition: 5 in a row
- MCTS simulations: 800 per move
- Neural network: 10 residual blocks, 128 channels
- Batch size: 64
- Learning rate: 0.001
- L2 regularization: 1e-4

## Performance

- Convergence: ~10-50 iterations
- Self-play parallelization: 4 workers
- GPU acceleration: CUDA if available

## References

- AlphaZero: Mastering Chess and Shogi by Self-Play (Silver et al., 2017)
- Gomoku rules: 15×15 board, 5 in a row wins