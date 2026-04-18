import subprocess
import torch


__all__ = ["get_best_gpu", "pick_device"]


def get_best_gpu(strategy='utilization'):
    if strategy == 'memory':
        free_mem = []
        for i in range(torch.cuda.device_count()):
            free_mem.append(torch.cuda.mem_get_info(i)[0])
        return free_mem.index(max(free_mem))

    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
        capture_output=True, text=True,
    )
    utilizations = [int(x.strip()) for x in result.stdout.strip().split('\n')]
    return utilizations.index(min(utilizations))


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        gpu_id = get_best_gpu(strategy='utilization')
        return torch.device(f'cuda:{gpu_id}')
    return torch.device('cpu')
