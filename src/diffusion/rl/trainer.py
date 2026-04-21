import time
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

from ..othello import OthelloEnv


__all__ = ["SelfPlayTrainer"]


class SelfPlayTrainer:
    """
    Owns the training loop and self-play data collection.

    All components (policy, verifier, optimizers, buffer) are injected
    by the caller — this class does not construct them.
    """

    def __init__(self, policy, verifier, policy_optim, verifier_optim, buffer, config):
        self.policy         = policy
        self.verifier       = verifier
        self.policy_optim   = policy_optim
        self.verifier_optim = verifier_optim
        self.buffer         = buffer
        self.config         = config
        self.device         = next(policy.parameters()).device

        self.current_iter = 0
        self.games_played = 0
        self.train_time_seconds = 0.0  # cumulative wall time across resumes

        total_steps = config['num_iterations'] * config['steps_per_iter']
        self.policy_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.policy_optim, T_max=total_steps, eta_min=config.get('eta_min', 1e-6)
        )
        self.verifier_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.verifier_optim, T_max=total_steps, eta_min=config.get('eta_min', 1e-6)
        )

        self.loss_history_policy   = []
        self.loss_history_verifier = []
        self.winrate_history       = []  # list of (iteration, win_rate)

    # ------------------------------------------------------------------ #
    # action selection                                                     #
    # ------------------------------------------------------------------ #

    def _random_action(self, state, valid_mask):
        valid_idxs = np.where(valid_mask)[0]
        return int(np.random.choice(valid_idxs))

    def _learned_action(self, state, valid_mask, temperature=0.0):
        self.policy.eval()
        return self.policy.select_action(state, valid_mask, temperature=temperature)

    def _training_action_fn(self):
        """
        Build a stateful action function for one self-play game that applies
        the AlphaZero-style temperature schedule: temp>0 for the first
        temp_drop_move plies, then argmax.
        """
        temp  = self.config.get('selfplay_temperature', 1.0)
        drop  = self.config.get('temp_drop_move', 15)
        moves = [0]

        def fn(state, valid_mask):
            t = temp if moves[0] < drop else 0.0
            moves[0] += 1
            return self._learned_action(state, valid_mask, temperature=t)
        return fn

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
        self.games_played += 1
        fn = self._training_action_fn()
        env, states, actions, outcomes = self.play_game(fn)
        return env, states, actions, outcomes

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

        mode = self.config.get('advantage_mode', 'none')
        if mode == 'verifier':
            self.verifier.eval()
            with torch.no_grad():
                baseline = self.verifier(z_t, states, t)
            adv = outcomes - baseline
            clip = self.config.get('advantage_clip', 1.0)
            if clip and clip > 0:
                adv = torch.clamp(adv, -clip, clip)
            weights = adv
        elif mode == 'raw':
            weights = outcomes
        else:
            weights = (outcomes + 1.0) / 2.0

        per_sample_loss = ((eps_pred - noise) ** 2).mean(dim=-1)
        loss            = (weights * per_sample_loss).mean()

        self.policy_optim.zero_grad()
        loss.backward()
        self.policy_optim.step()
        self.policy_scheduler.step()
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
        self.verifier_scheduler.step()
        return loss.item()

    # ------------------------------------------------------------------ #
    # main loop                                                            #
    # ------------------------------------------------------------------ #

    def train(self):
        cfg      = self.config
        ckpt_dir = Path(cfg['checkpoint_dir'])
        ckpt_dir.mkdir(exist_ok=True)

        print(f"Training on device: {self.device}")
        print(f"Advantage mode: {cfg.get('advantage_mode', 'none')}  "
              f"self-play temp: {cfg.get('selfplay_temperature', 1.0)} "
              f"(drops after move {cfg.get('temp_drop_move', 15)})")
        print(f"LR schedule: cosine {cfg['lr']:.2e} → {cfg.get('eta_min', 1e-6):.2e} "
              f"over {cfg['num_iterations']*cfg['steps_per_iter']:,} steps")
        print(f"Starting at iteration {self.current_iter}/{cfg['num_iterations']}  "
              f"prior train time: {self._fmt_time(self.train_time_seconds)}\n", flush=True)

        onnx_every = cfg.get('onnx_every', 0)

        while self.current_iter < cfg['num_iterations']:
            iter_t0 = time.time()
            self.current_iter += 1

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
                env, states, actions, outcomes = self.play_training_game()
                self.buffer.add_game(states, actions, outcomes)

            iter_sec = time.time() - iter_t0
            self.train_time_seconds += iter_sec
            cur_lr = self.policy_optim.param_groups[0]['lr']

            log = (f"iter {self.current_iter:3d}/{cfg['num_iterations']}"
                   f"  p_loss={avg_pl:.4f}"
                   f"  v_loss={avg_vl:.4f}"
                   f"  lr={cur_lr:.2e}"
                   f"  buffer={len(self.buffer):,}"
                   f"  dt={iter_sec:.1f}s"
                   f"  total={self._fmt_time(self.train_time_seconds)}")

            if self.current_iter % cfg['eval_every'] == 0:
                eval_out = self.evaluate_vs_random(cfg['eval_games'], diagnostics=True)
                w, d, l, diag = eval_out
                self.winrate_history.append((self.current_iter, w))
                log += (f"  vs_random W={w:.0%} (p0={diag['win_p0']:.0%} p1={diag['win_p1']:.0%})"
                        f" D={d:.0%} L={l:.0%}  H={diag['entropy_mean']:.2f}"
                        f" [{diag['entropy_min']:.2f},{diag['entropy_max']:.2f}]")
                print(log, flush=True)
                self._print_heatmap(diag['heatmap'])
            else:
                print(log, flush=True)

            self.save_checkpoint(ckpt_dir / 'latest.pt')
            if self.current_iter % cfg['checkpoint_every'] == 0:
                self.save_checkpoint(ckpt_dir / f'iter_{self.current_iter:04d}.pt')
                print(f"  ↳ saved iter_{self.current_iter:04d}.pt", flush=True)

            if onnx_every and self.current_iter % onnx_every == 0:
                self.save(
                    denoiser_path=str(ckpt_dir / 'denoiser.onnx'),
                    verifier_path=str(ckpt_dir / 'verifier.onnx'),
                )

        print(f"\nTotal training time: {self._fmt_time(self.train_time_seconds)} "
              f"({self.train_time_seconds:.1f}s)", flush=True)

    @staticmethod
    def _fmt_time(sec):
        sec = int(sec)
        h, rem = divmod(sec, 3600)
        m, s = divmod(rem, 60)
        if h:
            return f"{h}h{m:02d}m{s:02d}s"
        if m:
            return f"{m}m{s:02d}s"
        return f"{s}s"

    @staticmethod
    def _print_heatmap(heat):
        import numpy as np
        total = heat.sum()
        if total <= 0:
            return
        pct = heat / total * 100.0
        print("  action heatmap (% of moves):")
        for row in pct:
            print("   " + "  ".join(f"{v:4.1f}" for v in row))

    # ------------------------------------------------------------------ #
    # evaluation                                                           #
    # ------------------------------------------------------------------ #

    def evaluate_vs_random(self, n_games=100, diagnostics=False):
        """
        Play n_games vs a random opponent, alternating colors.

        If diagnostics=True, also tracks action entropy, per-cell heatmap,
        and win rate by color. Returns (w, d, l, diag_dict) in that case.
        """
        rows, cols = self.config['state_shape'][1], self.config['state_shape'][2]

        wins = draws = losses = 0
        w_p0 = g_p0 = 0
        w_p1 = g_p1 = 0
        heatmap = np.zeros((rows, cols), dtype=np.float64)
        entropies = []

        def agent_fn(state, valid_mask):
            if diagnostics:
                self.policy.eval()
                action, logits = self.policy.select_action(
                    state, valid_mask, temperature=0.0, return_logits=True
                )
                valid = torch.as_tensor(valid_mask, dtype=torch.bool)
                masked = logits[valid]
                if masked.numel() > 0:
                    probs = torch.softmax(masked, dim=-1)
                    eps = 1e-12
                    H = -(probs * (probs + eps).log()).sum().item()
                    entropies.append(H)
                r, c = divmod(action, cols)
                heatmap[r, c] += 1
                return action
            return self._learned_action(state, valid_mask)

        for i in range(n_games):
            if i % 2 == 0:
                env, _, _, _ = self.play_game(agent_fn, self._random_action)
                agent = 0
                g_p0 += 1
            else:
                env, _, _, _ = self.play_game(self._random_action, agent_fn)
                agent = 1
                g_p1 += 1
            r = env.outcome_for(agent)
            if r > 0:
                wins += 1
                if agent == 0: w_p0 += 1
                else:          w_p1 += 1
            elif r < 0:
                losses += 1
            else:
                draws += 1

        w = wins / n_games
        d = draws / n_games
        l = losses / n_games
        if not diagnostics:
            return w, d, l

        diag = {
            'win_p0':      w_p0 / max(g_p0, 1),
            'win_p1':      w_p1 / max(g_p1, 1),
            'entropy_mean': float(np.mean(entropies)) if entropies else 0.0,
            'entropy_min':  float(np.min(entropies))  if entropies else 0.0,
            'entropy_max':  float(np.max(entropies))  if entropies else 0.0,
            'heatmap':     heatmap,
        }
        return w, d, l, diag

    def evaluation(self, plot=False):
        print("\n=== Evaluation ===")

        w, d, l = self.evaluate_vs_random(200)
        print(f"Policy vs random (200 games):  W={w:.1%}  D={d:.1%}  L={l:.1%}")

        if self.loss_history_policy:
            print(f"\nTraining summary ({self.current_iter} iterations):")
            print(f"  Final policy loss:   {self.loss_history_policy[-1]:.4f}")
            print(f"  Final verifier loss: {self.loss_history_verifier[-1]:.4f}")
            print(f"  Total training time: {self._fmt_time(self.train_time_seconds)} "
                  f"({self.train_time_seconds:.1f}s)")

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

        return w, d, l

    # ------------------------------------------------------------------ #
    # checkpointing                                                        #
    # ------------------------------------------------------------------ #

    def save_checkpoint(self, path):
        torch.save({
            'policy_state':             self.policy.state_dict(),
            'verifier_state':           self.verifier.state_dict(),
            'policy_optim_state':       self.policy_optim.state_dict(),
            'verifier_optim_state':     self.verifier_optim.state_dict(),
            'policy_scheduler_state':   self.policy_scheduler.state_dict(),
            'verifier_scheduler_state': self.verifier_scheduler.state_dict(),
            'iteration':                self.current_iter,
            'games_played':             self.games_played,
            'train_time_seconds':       self.train_time_seconds,
            'buffer':                   self.buffer.state_dict(max_save=50_000),
            'loss_history_policy':      self.loss_history_policy,
            'loss_history_verifier':    self.loss_history_verifier,
            'winrate_history':          self.winrate_history,
        }, path)

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        try:
            self.policy.load_state_dict(ckpt['policy_state'])
            self.verifier.load_state_dict(ckpt['verifier_state'])
            self.policy_optim.load_state_dict(ckpt['policy_optim_state'])
            self.verifier_optim.load_state_dict(ckpt['verifier_optim_state'])

            # Optimizer state tensors (exp_avg, exp_avg_sq, etc.) are restored
            # on whatever device they were saved on. map_location only covers
            # the top-level payload, so we must walk the per-param state and
            # move any tensors to the current device — otherwise the first
            # optim.step() on resume raises "expected to be on device X".
            self._optim_state_to_device(self.policy_optim)
            self._optim_state_to_device(self.verifier_optim)

            if 'policy_scheduler_state' in ckpt:
                self.policy_scheduler.load_state_dict(ckpt['policy_scheduler_state'])
            if 'verifier_scheduler_state' in ckpt:
                self.verifier_scheduler.load_state_dict(ckpt['verifier_scheduler_state'])

            self.current_iter  = ckpt['iteration']
            self.games_played  = ckpt.get('games_played', 0)
            self.train_time_seconds    = ckpt.get('train_time_seconds', 0.0)
            self.loss_history_policy   = ckpt.get('loss_history_policy', [])
            self.loss_history_verifier = ckpt.get('loss_history_verifier', [])
            self.winrate_history       = ckpt.get('winrate_history', [])
            self.buffer.load_state_dict(ckpt.get('buffer'))
            print(f"Loaded checkpoint: {path}  "
                  f"(iter {self.current_iter}, buffer={len(self.buffer):,}, "
                  f"prior train time: {self._fmt_time(self.train_time_seconds)})")
        except (RuntimeError, KeyError) as e:
            print(f"WARNING: checkpoint mismatch — starting fresh. ({e})")
            print(f"  Delete {path} to suppress this warning.")

    def _optim_state_to_device(self, optimizer):
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)

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

    def watch_game(self, opponent='random', delay=None, step_mode=False):
        if delay is None:
            delay = self.config['watch_delay']

        fn = self._learned_action

        if opponent == 'random':
            fn0, fn1 = fn, self._random_action
            label = ['● agent', '○ random']
        elif opponent == 'self':
            fn0, fn1 = fn, fn
            label = ['● agent-0', '○ agent-1']
        else:
            raise ValueError(f"Unknown opponent: '{opponent}'. Use 'random' or 'self'.")

        speed_hint = ("step mode — press Enter to advance" if step_mode
                      else f"delay={delay}s")

        print(f"\n{'='*50}")
        print(f"  Watching: {label[0]} vs {label[1]}")
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
            print(f"  Move {move_num:2d}: {label[mover]} → ({r},{c})")

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
