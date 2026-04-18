import numpy as np
import gymnasium as gym
from gymnasium import spaces


__all__ = ["OthelloEnv"]


class OthelloEnv(gym.Env):
    """
    8x8 Othello as a single-player gymnasium.Env.

    Observation: (3, 8, 8) float32
        channel 0 — current player's stones
        channel 1 — opponent's stones
        channel 2 — valid moves mask (1.0 where legal)

    Action: int in [0, 63]  (row * 8 + col)

    The board perspective flips automatically after each step so
    channel 0 is always "the player to move". The network never
    needs to know which color it is playing.

    info dict always contains:
        'valid_mask': (64,) bool array
        'current_player': 0 or 1
    """

    SIZE = 8
    NUM_CELLS = 64
    DIRS = [(-1, -1), (-1, 0), (-1, 1),
            (0,  -1),           (0,  1),
            (1,  -1),  (1,  0), (1,  1)]

    metadata = {'render_modes': ['human']}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(3, self.SIZE, self.SIZE), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.NUM_CELLS)
        self.board = None
        self.current_player = None
        self.done = None

    # ------------------------------------------------------------------ #
    # gymnasium interface                                                  #
    # ------------------------------------------------------------------ #

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros((self.SIZE, self.SIZE), dtype=np.int8)
        mid = self.SIZE // 2
        self.board[mid - 1, mid - 1] = 2
        self.board[mid,     mid    ] = 2
        self.board[mid - 1, mid    ] = 1
        self.board[mid,     mid - 1] = 1
        self.current_player = 0
        self.done = False
        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action):
        assert not self.done, "step() called on finished game"
        r, c = divmod(int(action), self.SIZE)
        flips = self._flips(r, c, self.current_player)
        assert flips, f"Illegal move {action} for player {self.current_player}"

        me = self.current_player + 1
        self.board[r, c] = me
        for fr, fc in flips:
            self.board[fr, fc] = me

        self.current_player = 1 - self.current_player

        if not self._has_valid_move(self.current_player):
            self.current_player = 1 - self.current_player
            if not self._has_valid_move(self.current_player):
                self.done = True

        if (self.board != 0).all():
            self.done = True

        obs = self._get_obs()
        info = self._get_info()
        return obs, 0.0, self.done, False, info

    def render(self):
        stones = {0: '.', 1: '●', 2: '○'}
        print("     " + "  ".join(str(i) for i in range(self.SIZE)))
        print("   +" + "---" * self.SIZE + "+")
        for r in range(self.SIZE):
            row = "  ".join(stones[int(self.board[r, c])] for c in range(self.SIZE))
            print(f" {r} | {row} |")
        print("   +" + "---" * self.SIZE + "+")
        p0, p1 = self._scores()
        mover = '●' if self.current_player == 0 else '○'
        status = 'GAME OVER' if self.done else f'to move: {mover}'
        print(f"   ● player_0={p0}  ○ player_1={p1}  {status}\n")

    def close(self):
        pass

    # ------------------------------------------------------------------ #
    # outcome query (call only when done=True)                            #
    # ------------------------------------------------------------------ #

    def outcome_for(self, player_idx):
        """Returns +1.0 (win), -1.0 (loss), 0.0 (draw) for player_idx."""
        assert self.done
        p0, p1 = self._scores()
        if p0 == p1:
            return 0.0
        winner = 0 if p0 > p1 else 1
        return 1.0 if player_idx == winner else -1.0

    # ------------------------------------------------------------------ #
    # internals                                                            #
    # ------------------------------------------------------------------ #

    def _flips(self, r, c, player_idx):
        if self.board[r, c] != 0:
            return []
        me = player_idx + 1
        opp = 2 - player_idx
        all_flips = []
        for dr, dc in self.DIRS:
            line = []
            rr, cc = r + dr, c + dc
            while 0 <= rr < self.SIZE and 0 <= cc < self.SIZE and self.board[rr, cc] == opp:
                line.append((rr, cc))
                rr += dr
                cc += dc
            if line and 0 <= rr < self.SIZE and 0 <= cc < self.SIZE and self.board[rr, cc] == me:
                all_flips.extend(line)
        return all_flips

    def _valid_mask(self, player_idx):
        mask = np.zeros(self.NUM_CELLS, dtype=bool)
        for idx in range(self.NUM_CELLS):
            r, c = divmod(idx, self.SIZE)
            if self._flips(r, c, player_idx):
                mask[idx] = True
        return mask

    def _has_valid_move(self, player_idx):
        for idx in range(self.NUM_CELLS):
            r, c = divmod(idx, self.SIZE)
            if self._flips(r, c, player_idx):
                return True
        return False

    def _scores(self):
        return int((self.board == 1).sum()), int((self.board == 2).sum())

    def _get_obs(self):
        me = self.current_player + 1
        opp = 2 - self.current_player
        obs = np.zeros((3, self.SIZE, self.SIZE), dtype=np.float32)
        obs[0] = (self.board == me).astype(np.float32)
        obs[1] = (self.board == opp).astype(np.float32)
        obs[2] = self._valid_mask(self.current_player).reshape(self.SIZE, self.SIZE).astype(np.float32)
        return obs

    def _get_info(self):
        return {
            'valid_mask': self._valid_mask(self.current_player),
            'current_player': self.current_player,
        }


if __name__ == '__main__':
    import time
    env = OthelloEnv()
    obs, info = env.reset()
    print(f"obs shape: {obs.shape}  action_space: {env.action_space}")
    env.render()

    moves = 0
    while not env.done:
        valid = np.where(info['valid_mask'])[0]
        action = int(np.random.choice(valid))
        obs, reward, done, _, info = env.step(action)
        moves += 1

    print(f"Game over in {moves} moves")
    env.render()
    print(f"P0 outcome: {env.outcome_for(0)}  P1 outcome: {env.outcome_for(1)}")
