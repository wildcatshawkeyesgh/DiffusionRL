import math


__all__ = ["MCTSNode"]


class MCTSNode:
    """
    Node in the denoising tree.
    Each node holds a partially-denoised latent z_t at timestep t.
    """

    __slots__ = ('z_t', 't', 'parent', 'children', 'visits', 'value_sum')

    def __init__(self, z_t, t, parent=None):
        self.z_t = z_t
        self.t = t
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value_sum = 0.0

    @property
    def value(self):
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits

    @property
    def is_terminal(self):
        return self.t <= 0

    def puct_score(self, child, c_puct):
        Q = child.value
        P = 1.0 / max(len(self.children), 1)
        U = c_puct * P * math.sqrt(self.visits) / (1 + child.visits)
        return Q + U

    def best_child(self, c_puct):
        return max(self.children, key=lambda c: self.puct_score(c, c_puct))

    def most_visited_child(self):
        return max(self.children, key=lambda c: c.visits)
