import time
import random
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

from ..othello import OthelloEnv


__all__ = ["SelfPlayTrainer"]


class SelfPlayTrainer:
    """
    Owns the training loop and self-play data collection.

    All components (policy, verifier, mcts, optimizers, buffer) are
    injected by the caller — this class does not construct them.
    """

    def __init__(self, policy, verifier, mcts, policy_optim, verifier_optim, buffer, config):
        self.policy         = policy
        self.verifier       = verifier
        self.mcts           = mcts
        self.policy_optim   = policy_optim
        self.verifier_optim = verifier_optim
        self.buffer         = buffer
        self.config         = config
        self.device         = next(policy.parameters()).device

        self.current_iter = 0
        self.games_played = 0

        self.loss_history_policy   = []
        self.loss_history_verifier = []
        self.winrate_history       = []  # list of (iteration, win_rate)

    # ------------------------------------------------------------------ #
    # MCTS curriculum                                                      #
    # ------------------------------------------------------------------ #

    def _mcts_fraction(self):
        progress = self.current_iter / max(self.config['num_iterations'], 1)
        start = self.config['mcts_fraction_start']
        end   = self.config['mcts_fraction_end']
        style = self.config['mcts_ramp_style']

        if style == 'step':
            if progress < 0.25:
                return start
            elif progress < 0.50:
                return start + (end - start) * 0.33
            elif progress < 0.75:
                return start + (end - start) * 0.66
            else:
                return end
        elif style == 'exponential':
            return start + (end - start) * (progress ** 2)
        else:
            return start + (end - start) * progress

    def _should_use_mcts(self):
        return random.random() < self._mcts_fraction()

    # ------------------------------------------------------------------ #
    # action selection                                                     #
    # ------------------------------------------------------------------ #

    def _random_action(self, state, valid_mask):
        valid_idxs = np.where(valid_mask)[0]
        return int(np.random.choice(valid_idxs))

    def _learned_action(self, state, valid_mask):
        self.policy.eval()
        return self.policy.select_action(state, valid_mask)

    def _mcts_action(self, state, valid_mask):
        return self.mcts.select_action(state, valid_mask)

    # ------------------------------------------------------------------ #
    # game playing                                                         #
    # ------------------------------------------------------------------ #

    def play_game(self, policy_fn_0, policy_fn_1=None):
        if policy_fn_1 is None:
            policy_fn_1 = policy_fn_0

        env = OthelloEnv()
        obs, info = env.reset()

        states, actions, movers = [], [], []

        while not env.done:
            valid_mask = info['valid_mask']
            if not valid_mask.any():
                break
            mover  = env.current_player
            fn     = policy_fn_0 if mover == 0 else policy_fn_1
            action = fn(obs, valid_mask)

            states.append(torch.as_tensor(obs, dtype=torch.float32))
            actions.append(action)
            movers.append(mover)

            obs, _, done, _, info = env.step(action)

        outcomes = [env.outcome_for(m) for m in movers]
        return env, states, actions, outcomes

    def play_training_game(self):
        use_mcts = self._should_use_mcts()
        self.games_played += 1
        fn = self._mcts_action if use_mcts else self._learned_action
        env, states, actions, outcomes = self.play_game(fn)
        return env, states, actions, outcomes, use_mcts

    # ------------------------------------------------------------------ #
    # bootstrap / data generation                                          #
    # ------------------------------------------------------------------ #

    def bootstrap_more(self, n):
        """
        Play n purely random games and append to the buffer.
        Safe to call at any time — before training, after loading a
        checkpoint, or mid-run to top off the buffer with fresh data.
        """
        print(f"Generating {n} random games...")
        t0 = time.time()
        for i in range(n):
            _, states, actions, outcomes = self.play_game(self._random_action)
            self.buffer.add_game(states, actions, outcomes)
            if (i + 1) % 500 == 0:
                elapsed = time.time() - t0
                print(f"  {i+1}/{n}  buffer={len(self.buffer):,}  ({elapsed:.1f}s)")
        print(f"Done in {time.time()-t0:.1f}s  buffer={len(self.buffer):,} triples\n")

    # ------------------------------------------------------------------ #
    # training steps                                                       #
    # ------------------------------------------------------------------ #

    def train_policy_step(self):
        if len(self.buffer) < self.config['batch_size']:
            return None

        states, actions, outcomes = self.buffer.sample(self.config['batch_size'])
        states   = states.to(self.device)
        actions  = actions.to(self.device)
        outcomes = outcomes.to(self.device)

        B   = actions.shape[0]
        z_0 = F.one_hot(actions, num_classes=self.config['action_dim']).float()
        t   = torch.randint(1, self.config['T'] + 1, (B,), device=self.device)

        z_t, noise = self.policy.forward_diffuse(z_0, t)

        self.policy.train()
        eps_pred = self.policy.denoiser(z_t, t, states)

        weights         = (outcomes + 1.0) / 2.0
        per_sample_loss = ((eps_pred - noise) ** 2).mean(dim=-1)
        loss            = (weights * per_sample_loss).mean()

        self.policy_optim.zero_grad()
        loss.backward()
        self.policy_optim.step()
        return loss.item()

    def train_verifier_step(self):
        if len(self.buffer) < self.config['batch_size']:
            return None

        states, actions, outcomes = self.buffer.sample(self.config['batch_size'])
        states   = states.to(self.device)
        actions  = actions.to(self.device)
        outcomes = outcomes.to(self.device)

        B   = actions.shape[0]
        z_0 = F.one_hot(actions, num_classes=self.config['action_dim']).float()
        t   = torch.randint(1, self.config['T'] + 1, (B,), device=self.device)

        z_t, _ = self.policy.forward_diffuse(z_0, t)

        self.verifier.train()
        pred = self.verifier(z_t.detach(), states, t)
        loss = F.mse_loss(pred, outcomes)

        self.verifier_optim.zero_grad()
        loss.backward()
        self.verifier_optim.step()
        return loss.item()

    # ------------------------------------------------------------------ #
    # main loop                                                            #
    # ------------------------------------------------------------------ #

    def train(self):
        cfg      = self.config
        ckpt_dir = Path(cfg['checkpoint_dir'])
        ckpt_dir.mkdir(exist_ok=True)

        print(f"Training on device: {self.device}")
        frac_s = cfg['mcts_fraction_start']
        frac_e = cfg['mcts_fraction_end']
        print(f"MCTS curriculum: {frac_s:.0%} → {frac_e:.0%} ({cfg['mcts_ramp_style']})")
        print(f"MCTS per action: {cfg['mcts_iterations']} iters, k={cfg['macro_step_k']}, "
              f"children={cfg['mcts_num_children']}, c_puct={cfg['puct_c']}")
        print(f"Starting at iteration {self.current_iter}/{cfg['num_iterations']}\n")

        while self.current_iter < cfg['num_iterations']:
            self.current_iter += 1
            mcts_games = 0

            p_losses, v_losses = [], []
            for _ in range(cfg['steps_per_iter']):
                pl = self.train_policy_step()
                vl = self.train_verifier_step()
                if pl is not None:
                    p_losses.append(pl)
                if vl is not None:
                    v_losses.append(vl)
            avg_pl = sum(p_losses) / max(len(p_losses), 1)
            avg_vl = sum(v_losses) / max(len(v_losses), 1)

            self.loss_history_policy.append(avg_pl)
            self.loss_history_verifier.append(avg_vl)

            for _ in range(cfg['games_per_iter']):
                env, states, actions, outcomes, used_mcts = self.play_training_game()
                self.buffer.add_game(states, actions, outcomes)
                if used_mcts:
                    mcts_games += 1

            log = (f"iter {self.current_iter:3d}/{cfg['num_iterations']}"
                   f"  p_loss={avg_pl:.4f}"
                   f"  v_loss={avg_vl:.4f}"
                   f"  mcts={mcts_games}/{cfg['games_per_iter']}"
                   f"  buffer={len(self.buffer):,}")

            if self.current_iter % cfg['eval_every'] == 0:
                w, d, l = self.evaluate_vs_random(cfg['eval_games'])
                self.winrate_history.append((self.current_iter, w))
                log += f"  vs_random W={w:.0%} D={d:.0%} L={l:.0%}"

            print(log)

            self.save_checkpoint(ckpt_dir / 'latest.pt')
            if self.current_iter % cfg['checkpoint_every'] == 0:
                self.save_checkpoint(ckpt_dir / f'iter_{self.current_iter:04d}.pt')
                print(f"  ↳ saved iter_{self.current_iter:04d}.pt")

    # ------------------------------------------------------------------ #
    # evaluation                                                           #
    # ------------------------------------------------------------------ #

    def evaluate_vs_random(self, n_games=100, use_mcts=False):
        wins = draws = losses = 0
        fn = self._mcts_action if use_mcts else self._learned_action
        for i in range(n_games):
            if i % 2 == 0:
                env, _, _, _ = self.play_game(fn, self._random_action)
                agent = 0
            else:
                env, _, _, _ = self.play_game(self._random_action, fn)
                agent = 1
            r = env.outcome_for(agent)
            if r > 0:
                wins += 1
            elif r < 0:
                losses += 1
            else:
                draws += 1
        return wins / n_games, draws / n_games, losses / n_games

    def evaluation(self, plot=False):
        print("\n=== Evaluation ===")

        w, d, l = self.evaluate_vs_random(200, use_mcts=False)
        print(f"Policy vs random (200 games):  W={w:.1%}  D={d:.1%}  L={l:.1%}")

        w_m, d_m, l_m = self.evaluate_vs_random(50, use_mcts=True)
        print(f"MCTS   vs random  (50 games):  W={w_m:.1%}  D={d_m:.1%}  L={l_m:.1%}")

        if self.loss_history_policy:
            print(f"\nTraining summary ({self.current_iter} iterations):")
            print(f"  Final policy loss:   {self.loss_history_policy[-1]:.4f}")
            print(f"  Final verifier loss: {self.loss_history_verifier[-1]:.4f}")

        if plot:
            import matplotlib.pyplot as plt

            if self.loss_history_policy:
                plt.figure()
                plt.plot(self.loss_history_policy,   label='policy')
                plt.plot(self.loss_history_verifier, label='verifier')
                plt.title('Training Loss')
                plt.xlabel('Iteration')
                plt.ylabel('Loss')
                plt.legend()
                plt.savefig('loss_curve.png')
                plt.close()
                print("Saved loss_curve.png")

            if self.winrate_history:
                iters, rates = zip(*self.winrate_history)
                plt.figure()
                plt.plot(iters, rates)
                plt.title('Win Rate vs Random')
                plt.xlabel('Iteration')
                plt.ylabel('Win Rate')
                plt.ylim(0, 1)
                plt.savefig('winrate_curve.png')
                plt.close()
                print("Saved winrate_curve.png")

        return w, d, l, w_m, d_m, l_m

    # ------------------------------------------------------------------ #
    # checkpointing                                                        #
    # ------------------------------------------------------------------ #

    def save_checkpoint(self, path):
        torch.save({
            'policy_state':         self.policy.state_dict(),
            'verifier_state':       self.verifier.state_dict(),
            'policy_optim_state':   self.policy_optim.state_dict(),
            'verifier_optim_state': self.verifier_optim.state_dict(),
            'iteration':            self.current_iter,
            'games_played':         self.games_played,
            'buffer':               self.buffer.state_dict(max_save=50_000),
            'loss_history_policy':  self.loss_history_policy,
            'loss_history_verifier': self.loss_history_verifier,
            'winrate_history':      self.winrate_history,
        }, path)

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        try:
            self.policy.load_state_dict(ckpt['policy_state'])
            self.verifier.load_state_dict(ckpt['verifier_state'])
            self.policy_optim.load_state_dict(ckpt['policy_optim_state'])
            self.verifier_optim.load_state_dict(ckpt['verifier_optim_state'])
            self.current_iter  = ckpt['iteration']
            self.games_played  = ckpt.get('games_played', 0)
            self.loss_history_policy   = ckpt.get('loss_history_policy', [])
            self.loss_history_verifier = ckpt.get('loss_history_verifier', [])
            self.winrate_history       = ckpt.get('winrate_history', [])
            self.buffer.load_state_dict(ckpt.get('buffer'))
            print(f"Loaded checkpoint: {path}  "
                  f"(iter {self.current_iter}, buffer={len(self.buffer):,})")
        except (RuntimeError, KeyError) as e:
            print(f"WARNING: checkpoint mismatch — starting fresh. ({e})")
            print(f"  Delete {path} to suppress this warning.")

    # ------------------------------------------------------------------ #
    # ONNX export                                                          #
    # ------------------------------------------------------------------ #

    def save(self, denoiser_path='denoiser.onnx', verifier_path='verifier.onnx'):
        """Export the two networks needed to play a game."""
        cfg    = self.config
        device = self.device

        self.policy.eval()
        self.verifier.eval()

        z_dummy     = torch.zeros(1, cfg['action_dim'], device=device)
        t_dummy     = torch.zeros(1, dtype=torch.long, device=device)
        state_dummy = torch.zeros(1, *cfg['state_shape'], device=device)

        torch.onnx.export(
            self.policy.denoiser,
            (z_dummy, t_dummy, state_dummy),
            denoiser_path,
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=['z_t', 't', 'board_state'],
            output_names=['noise_pred'],
            dynamic_axes={
                'z_t':         {0: 'batch'},
                't':           {0: 'batch'},
                'board_state': {0: 'batch'},
                'noise_pred':  {0: 'batch'},
            },
        )
        print(f"Saved denoiser → {denoiser_path}")

        torch.onnx.export(
            self.verifier,
            (z_dummy, state_dummy, t_dummy),
            verifier_path,
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=['z_t', 'board_state', 't'],
            output_names=['value'],
            dynamic_axes={
                'z_t':         {0: 'batch'},
                'board_state': {0: 'batch'},
                't':           {0: 'batch'},
                'value':       {0: 'batch'},
            },
        )
        print(f"Saved verifier  → {verifier_path}")

    # ------------------------------------------------------------------ #
    # watch mode                                                           #
    # ------------------------------------------------------------------ #

    def watch_game(self, opponent='random', use_mcts=False, delay=None, step_mode=False):
        if delay is None:
            delay = self.config['watch_delay']

        fn = self._mcts_action if use_mcts else self._learned_action

        if opponent == 'random':
            fn0, fn1 = fn, self._random_action
            label = ['● agent', '○ random']
        elif opponent == 'self':
            fn0, fn1 = fn, fn
            label = ['● agent-0', '○ agent-1']
        else:
            raise ValueError(f"Unknown opponent: '{opponent}'. Use 'random' or 'self'.")

        mcts_tag   = " (MCTS)" if use_mcts else ""
        speed_hint = ("step mode — press Enter to advance" if step_mode
                      else f"delay={delay}s")

        print(f"\n{'='*50}")
        print(f"  Watching: {label[0]}{mcts_tag} vs {label[1]}")
        print(f"  {speed_hint}")
        print(f"{'='*50}\n")

        env = OthelloEnv()
        obs, info = env.reset()
        env.render()

        move_num = 0
        while not env.done:
            valid_mask = info['valid_mask']
            if not valid_mask.any():
                break

            mover  = env.current_player
            f      = fn0 if mover == 0 else fn1
            action = f(obs, valid_mask)

            move_num += 1
            r, c = divmod(action, env.SIZE)
            print(f"  Move {move_num:2d}: {label[mover]}{mcts_tag} → ({r},{c})")

            obs, _, done, _, info = env.step(action)
            env.render()

            if step_mode:
                input("  [Enter to continue] ")
            elif delay > 0:
                time.sleep(delay)

        p0, p1 = env._scores()
        print(f"{'='*50}")
        print(f"  Final: ●={p0}  ○={p1}  — ", end='')
        r0 = env.outcome_for(0)
        if r0 > 0:
            print(f"{label[0]} wins!")
        elif r0 < 0:
            print(f"{label[1]} wins!")
        else:
            print("Draw.")
        print(f"{'='*50}\n")
