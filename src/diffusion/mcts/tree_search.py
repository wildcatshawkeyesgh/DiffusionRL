import torch

from .node import MCTSNode


__all__ = ["TreeDiffMCTS"]


class TreeDiffMCTS:
    """
    MCTS over denoising trajectories.

    Expands nodes by running macro-steps of the diffusion policy
    (k denoising steps per expansion). Evaluates leaves with the
    verifier instead of game rollouts.
    """

    def __init__(self, diffusion_policy, verifier, config):
        self.policy = diffusion_policy
        self.verifier = verifier
        self.device = next(diffusion_policy.parameters()).device

        self.T = config['T']
        self.action_dim = config['action_dim']
        self.num_iterations = config['mcts_iterations']
        self.macro_step_k = config['macro_step_k']
        self.c_puct = config['puct_c']
        self.num_children = config['mcts_num_children']

    @torch.no_grad()
    def select_action(self, board_state_np, valid_mask_np):
        """
        Run MCTS over denoising trajectories and return best action.

        board_state_np: (3, 8, 8) numpy array
        valid_mask_np:  (64,) bool numpy array
        returns: int action
        """
        self.policy.eval()
        self.verifier.eval()

        board = torch.as_tensor(
            board_state_np, dtype=torch.float32
        ).unsqueeze(0).to(self.device)
        valid = torch.as_tensor(valid_mask_np, dtype=torch.bool)

        z_T = torch.randn(self.action_dim, device=self.device)
        root = MCTSNode(z_t=z_T, t=self.T)

        for _ in range(self.num_iterations):
            node = self._select(root)

            if node.is_terminal:
                value = self._evaluate(node, board)
                self._backprop(node, value)
            else:
                children = self._expand(node, board)
                for child in children:
                    value = self._evaluate(child, board)
                    self._backprop(child, value)

        best = self._best_terminal(root, board)
        logits = best.z_t.cpu()
        logits[~valid] = float('-inf')
        return int(torch.argmax(logits).item())

    def _select(self, root):
        node = root
        while node.children and not node.is_terminal:
            node = node.best_child(self.c_puct)
        return node

    def _expand(self, node, board):
        children = []
        for _ in range(self.num_children):
            z = node.z_t.clone()
            t_cur = node.t

            for _ in range(self.macro_step_k):
                if t_cur <= 0:
                    break
                t_tensor = torch.full((1,), t_cur, dtype=torch.long, device=self.device)
                z_batch = z.unsqueeze(0)
                z_batch = self.policy.denoise_step(z_batch, t_tensor, board)
                z = z_batch.squeeze(0)
                t_cur -= 1

            child = MCTSNode(z_t=z, t=t_cur, parent=node)
            node.children.append(child)
            children.append(child)
        return children

    def _evaluate(self, node, board):
        z = node.z_t.unsqueeze(0)
        t = torch.full((1,), max(node.t, 0), dtype=torch.long, device=self.device)
        return self.verifier(z, board, t).item()

    def _backprop(self, node, value):
        cur = node
        while cur is not None:
            cur.visits += 1
            cur.value_sum += value
            cur = cur.parent

    def _best_terminal(self, root, board):
        """Descend via most-visited, finishing denoising if needed."""
        node = root
        while node.children:
            node = node.most_visited_child()

        if not node.is_terminal:
            z = node.z_t.clone()
            t_cur = node.t
            while t_cur > 0:
                t_tensor = torch.full((1,), t_cur, dtype=torch.long, device=self.device)
                z_batch = z.unsqueeze(0)
                z_batch = self.policy.denoise_step(z_batch, t_tensor, board)
                z = z_batch.squeeze(0)
                t_cur -= 1
            node = MCTSNode(z_t=z, t=0)

        return node
