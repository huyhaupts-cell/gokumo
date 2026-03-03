import numpy as np
import torch

class Node:
    def __init__(self, board, player, parent=None, prior=0.0):
        self.board = board
        self.player = player
        self.parent = parent

        self.children = {}
        self.prior = prior

        self.visit_count = 0
        self.total_value = 0.0

    @property
    def q(self):
        if self.visit_count == 0:
            return 0
        return self.total_value / self.visit_count

    def ucb(self, c_puct):
        return self.q + c_puct * self.prior * (
            np.sqrt(self.parent.visit_count) / (1 + self.visit_count)
        )


class MCTS:
    def __init__(self, env, network, num_simulations=400, c_puct=1.25):
        self.env = env
        self.network = network
        self.num_simulations = num_simulations
        self.c_puct = c_puct

    def search(self, root_board, root_player):

        root = Node(root_board, root_player)

        # Evaluate root
        policy, value = self._evaluate_batch([root])
        self._expand(root, policy[0])
        root.visit_count += 1
        root.total_value += value[0]

        for _ in range(self.num_simulations):

            node = root
            search_path = [node]

            # Selection
            while node.children:
                node = max(node.children.values(),
                           key=lambda n: n.ucb(self.c_puct))
                search_path.append(node)

            # Evaluate leaf
            policy, value = self._evaluate_batch([node])

            # Expand
            self._expand(node, policy[0])

            # Backup
            self._backup(search_path, value[0])

        return self._get_action_probs(root)

    def _evaluate_batch(self, nodes):
        boards = [n.board for n in nodes]
        players = [n.player for n in nodes]

        input_tensor = self.network.prepare_batch(boards, players)

        with torch.no_grad():
            policy, value = self.network(input_tensor)

        return policy.cpu().numpy(), value.squeeze().cpu().numpy()

    def _expand(self, node, policy):

        valid_moves = np.where(node.board.flatten() == 0)[0]

        policy = np.exp(policy)
        masked_policy = np.zeros_like(policy)
        masked_policy[valid_moves] = policy[valid_moves]

        masked_policy /= np.sum(masked_policy)

        # Dirichlet noise at root
        if node.parent is None:
            noise = np.random.dirichlet([0.03] * len(valid_moves))
            masked_policy[valid_moves] = (
                0.75 * masked_policy[valid_moves] + 0.25 * noise
            )

        for move in valid_moves:
            new_board = node.board.copy()
            x, y = divmod(move, self.env.board_size)
            new_board[x, y] = node.player

            node.children[move] = Node(
                new_board,
                3 - node.player,
                parent=node,
                prior=masked_policy[move]
            )

    def _backup(self, path, value):
        for node in reversed(path):
            node.visit_count += 1
            node.total_value += value
            value = -value

    def _get_action_probs(self, root):
        action_probs = np.zeros(self.env.board_size ** 2)

        for move, child in root.children.items():
            action_probs[move] = child.visit_count

        action_probs /= np.sum(action_probs)
        return action_probs, root.q