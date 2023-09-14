import torch.optim.optimizer as Optimizer
import torch

class Lookahead(Optimizer):
    """
    Lookahead Optimization as proposed by https://arxiv.org/abs/1907.08610.
    This is a wrapper class that can be applied to an instantiated optimizer.
    """
    def __init__(self, optimizer, k=5, alpha=0.5):
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.param_groups = optimizer.param_groups

        self.counter = 0
        for group in optimizer.param_groups:
            group['phi'] = []
            for param in group['params']:
                phi_param = torch.ones_like(param).mul_(param.data)
                group['phi'].append(phi_param)

    def step(self):
        if self.counter == self.k:
            for group_idx, group in enumerate(self.param_groups):
                for idx, _ in enumerate(group['phi']):
                    theta = self.optimizer.param_groups[group_idx]['params'][idx].data
                    group['phi'][idx] = group['phi'][idx] + self.alpha * (theta - group['phi'][idx])
            self.counter = 0
        else:
            self.counter += 1
            self.optimizer.step()