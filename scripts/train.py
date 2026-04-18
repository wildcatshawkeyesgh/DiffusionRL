import yaml
import torch
from pathlib import Path

from diffusion import models, rl, mcts, utils


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(path=None):
    if path is None:
        path = PROJECT_ROOT / 'config.yaml'
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg['state_shape'] = tuple(cfg['state_shape'])
    return cfg


def main():
    cfg = load_config()
    torch.manual_seed(cfg['seed'])

    device = utils.pick_device()
    cfg['device'] = str(device)

    print(f"Device : {device}")
    print(f"Board  : {cfg['state_shape']}")
    print(f"Actions: {cfg['action_dim']}")
    print(f"T      : {cfg['T']}  hidden: {cfg['hidden_dim']}")
    print()

    policy         = models.DiffusionPolicy(cfg).to(device)
    verifier       = models.DualSpaceVerifier(cfg).to(device)
    tree           = mcts.TreeDiffMCTS(policy, verifier, cfg)
    policy_optim   = torch.optim.Adam(policy.parameters(),   lr=cfg['lr'])
    verifier_optim = torch.optim.Adam(verifier.parameters(), lr=cfg['lr'])
    buffer         = rl.ReplayBuffer(cfg['buffer_max_triples'])

    trainer = rl.SelfPlayTrainer(
        policy, verifier, tree,
        policy_optim, verifier_optim,
        buffer, cfg,
    )

    ckpt = Path(cfg['checkpoint_dir']) / 'latest.pt'
    if ckpt.exists():
        trainer.load_checkpoint(ckpt)
        if len(buffer) == 0:
            print("Buffer empty after checkpoint load — running bootstrap.")
            trainer.bootstrap_more(cfg['num_bootstrap_games'])
    else:
        trainer.bootstrap_more(cfg['num_bootstrap_games'])

    trainer.train()
    trainer.save()
    trainer.evaluation(plot=cfg.get('save_plots', False))

    print()
    trainer.watch_game(opponent='random', delay=0.3)


if __name__ == '__main__':
    main()
