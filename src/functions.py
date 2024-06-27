import torch
from torch import Tensor


def cross_correlation(X: Tensor, K: Tensor) -> Tensor:
    n, m = X.shape
    r, s = K.shape
    a, b = n - r + 1, m - s + 1
    Y: Tensor = torch.zeros((a, b))
    for row_idx in range(a):
        for col_idx in range(b):
            prod: Tensor = X[row_idx : row_idx + r, col_idx : col_idx + s] * K
            Y[row_idx][col_idx] = prod.sum()
    return Y
