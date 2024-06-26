from abc import ABC, abstractmethod

import torch
from torch import Tensor


class Module(ABC):
    @abstractmethod
    def forward(self, X) -> Tensor: ...

    def __call__(self, X) -> Tensor:
        return self.forward(X)


class Linear(Module):
    def __init__(self, n_in: int, n_out: int, sigma: float = 0.01):
        self.weight = torch.normal(0, sigma, (n_in, 1), requires_grad=True)
        self.bias = torch.zeros(1, requires_grad=True)

    def forward(self, X) -> Tensor:
        return X @ self.weight + self.bias


class Loss(ABC):
    @abstractmethod
    def forward(self, X, y) -> Tensor: ...

    def __call__(self, X, y) -> Tensor:
        return self.forward(X, y)


class MSELoss(Loss):
    def forward(self, X, y) -> Tensor:
        return ((X - y) ** 2 / 2).sum()


class MSEMeanLoss(Loss):
    def forward(self, X, y) -> Tensor:
        return ((X - y) ** 2 / 2.0).mean()


class Optimizer(ABC):
    def __init__(self, params: list[Tensor], lr: float = 0.01):
        self.initialize_parameters()
        self.lr = lr

    @abstractmethod
    def step(self): ...

    @property
    def params_with_grad(self) -> list[Tensor]:
        return [param for param in self.params if param.grad is not None]

    def zero_grad(self) -> None:
        for param in self.params_with_grad:
            param.grad.zero_()


class SGD(Optimizer):
    def step(self):
        for param in self.params_with_grad:
            param.data -= param.grad * self.lr

    def initialize_parameters(self, w, b) -> None:
        self.params = [w, b]


class Trainer:
    def __init__(
        self, model: Module, loss_function: Loss, optimizer: Optimizer, lr: float = 0.01
    ):
        self.model: Module = model
        self.loss_function: Loss = loss_function
        self.lr = lr

    @abstractmethod
    def initialize_optimizer(): ...

    def loss(self, X, y) -> Tensor:
        return self.loss_function(self.model(X), y)


class LinearRegression(Trainer):
    def initialize_optimizer(self, w, b):
        self.optimizer = SGD([w, b], lr=self.lr)

    def train(self, X, y, epochs: int):
        w = torch.normal(0, 0.01, (X.shape[1], 1), requires_grad=True)
        b = torch.zeros(1, requires_grad=True)
        self.initialize_optimizer(w, b)

        for epoch in range(epochs):
            self.optimizer.zero_grad()
            loss = self.loss(X, y)
            loss.backward()
            self.optimizer.step()
            print(f"Epoch {epoch}: loss {loss.item()}")
