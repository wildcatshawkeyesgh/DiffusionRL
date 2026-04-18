import random
import torch


__all__ = ["ReplayBuffer"]


class ReplayBuffer:
    """
    Circular buffer of (state, action, outcome) triples.

    Stores flattened per-move records so sampling is O(1).
    State tensors stay as float32 CPU tensors; moved to device at
    train time inside the trainer.
    """

    def __init__(self, max_triples):
        self.max = max_triples
        self.states   = [None] * max_triples
        self.actions  = [None] * max_triples
        self.outcomes = [None] * max_triples
        self.pos  = 0
        self.size = 0

    def add_game(self, states, actions, outcomes):
        """
        states:   list of (3, 8, 8) float32 tensors
        actions:  list of int
        outcomes: list of float  (per move, from that mover's POV)
        """
        for s, a, o in zip(states, actions, outcomes):
            self.states[self.pos]   = s
            self.actions[self.pos]  = a
            self.outcomes[self.pos] = o
            self.pos  = (self.pos + 1) % self.max
            self.size = min(self.size + 1, self.max)

    def sample(self, batch_size):
        """Returns (states, actions, outcomes) tensors on CPU."""
        idxs     = random.choices(range(self.size), k=batch_size)
        states   = torch.stack([self.states[i] for i in idxs])
        actions  = torch.tensor([self.actions[i] for i in idxs], dtype=torch.long)
        outcomes = torch.tensor([self.outcomes[i] for i in idxs], dtype=torch.float32)
        return states, actions, outcomes

    def state_dict(self, max_save=50_000):
        """Serialize the most recent min(size, max_save) entries."""
        n = min(self.size, max_save)
        if n == 0:
            return None

        if self.size < self.max:
            # Buffer not yet full — valid entries are 0..size-1, save last n
            start   = max(0, self.size - n)
            indices = list(range(start, self.size))
        else:
            # Full circular buffer — most recent n entries end at pos-1
            indices = [(self.pos - n + i) % self.max for i in range(n)]

        states   = torch.stack([self.states[i] for i in indices])
        actions  = torch.tensor([self.actions[i] for i in indices], dtype=torch.long)
        outcomes = torch.tensor([self.outcomes[i] for i in indices], dtype=torch.float32)
        return {'states': states, 'actions': actions, 'outcomes': outcomes}

    def load_state_dict(self, d):
        """Restore entries from a state_dict produced by state_dict()."""
        if d is None:
            return
        states   = d['states']    # (N, 3, 8, 8)
        actions  = d['actions']   # (N,)
        outcomes = d['outcomes']  # (N,)
        n = len(states)
        for i in range(n):
            self.states[self.pos]   = states[i]
            self.actions[self.pos]  = actions[i].item()
            self.outcomes[self.pos] = outcomes[i].item()
            self.pos  = (self.pos + 1) % self.max
            self.size = min(self.size + 1, self.max)

    def __len__(self):
        return self.size
