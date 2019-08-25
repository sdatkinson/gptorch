# File: mean_functions.py
# File Created: Saturday, 13th July 2019 3:42:16 pm
# Author: Steven Atkinson (steven@atkinson.mn)

"""
mean_functions.py: A few extra mean functions that you don't usually see as
PyTorch modules
"""

import torch

from .util import torch_dtype

torch.set_default_dtype(torch_dtype)


class Constant(torch.nn.Module):
    """
    Constant mean function
    """

    def __init__(self, dy: int, val: torch.Tensor = None):
        super().__init__()
        if val is not None:
            if not val.shape[0] == dy:
                raise ValueError("Provided val doesn't match output dimension")
            val = val.clone()
        else:
            val = torch.zeros(dy)

        self._dy = dy
        self.val = torch.nn.Parameter(val)

    def forward(self, x):
        return torch.zeros(x.shape[0], self._dy) + self.val


class Zero(Constant):
    """
    Zero mean function (default for GPs).
    """

    def __init__(self, dy: int):
        super().__init__(dy)
        self.val.requires_grad_(False)
