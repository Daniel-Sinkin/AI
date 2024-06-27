import torch
from torch import Tensor

from .functions import cross_correlation


class Conv2D:
    def __init__(self, kernel_size: int | tuple[int, int]):
        if isinstance(kernel_size, int):
            self.weight: Tensor = torch.rand((kernel_size, kernel_size))
        else:
            assert isinstance(kernel_size, tuple)
            assert len(kernel_size) == 2
            assert all(isinstance(x, int) for x in kernel_size)
            self.weight: Tensor = torch.rand(kernel_size)
        self.bias: Tensor = torch.zeros(1)

    def forward(self, X) -> Tensor:
        return cross_correlation(X, self.weight) + self.bias

    def __call__(self, X) -> Tensor:
        return self.forward(X)
