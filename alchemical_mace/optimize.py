import torch
from typing import Union, Iterable, Dict, Any


class ExponentiatedGradientDescent(torch.optim.Optimizer):
    """
    Implements Exponentiated Gradient Descent.

    Args:
        params (iterable of torch.Tensor or dict): iterable of parameters to optimize or
            dicts defining parameter groups.
        lr (float, optional): learning rate. Defaults to 1e-3.
        eps (float, optional): small constant for numerical stability. Defaults to 1e-8.
    """

    def __init__(
        self,
        params: Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]],
        lr: float = 1e-3,
        eps: float = 1e-8,
    ):
        super().__init__(params, defaults=dict(lr=lr, eps=eps))

    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data.mul_(torch.exp(-group["lr"] * p.grad))
                p.data.div_(p.data.sum() + group["eps"])
