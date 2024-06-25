from typing import Tuple, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import make_moons
from torch import nn, optim


def prepare_data(
    N: int, train_pct: float, seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    _rng = np.random.default_rng(seed)
    N_train = int(N * train_pct)
    N_test = (N - N_train) // 2
    N_valid = N - N_train - N_test
    assert N == N_train + N_test + N_valid

    idxs = np.arange(N)
    _rng.shuffle(idxs)

    data, labels = make_moons(n_samples=N, noise=0.2, random_state=seed)
    data_shuffled = cast(np.ndarray, data[idxs])
    labels_shuffled = cast(np.ndarray, labels[idxs])

    data_train = cast(np.ndarray, data_shuffled[:N_train])
    labels_train = cast(np.ndarray, labels_shuffled[:N_train])
    data_test = cast(np.ndarray, data_shuffled[N_train : N_train + N_test])
    labels_test = cast(np.ndarray, labels_shuffled[N_train + N_test :])
    data_valid = cast(np.ndarray, data_shuffled[N_train + N_test :])
    labels_valid = cast(np.ndarray, labels_shuffled[N_train + N_test :])

    return data_train, labels_train, data_test, labels_test, data_valid, labels_valid


def plot_data(data: np.ndarray, labels: np.ndarray) -> None:
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap=plt.cm.viridis)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Moons Dataset")
    plt.show()


def create_model() -> nn.Sequential:
    model = nn.Sequential()
    model.add_module("linear1", nn.Linear(2, 6))
    model.add_module("relu1", nn.ReLU())
    model.add_module("linear2", nn.Linear(6, 20))
    model.add_module("relu2", nn.ReLU())
    model.add_module("linear3", nn.Linear(20, 20))
    model.add_module("relu3", nn.ReLU())
    model.add_module("linear4", nn.Linear(20, 6))
    model.add_module("relu4", nn.ReLU())
    model.add_module("linear5", nn.Linear(6, 1))
    model.add_module("sigmoid", nn.Sigmoid())

    # Initialize weights
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    return model


def plot_decision_boundary(
    model: nn.Sequential, data: np.ndarray, labels: np.ndarray, epoch: int
) -> None:
    x_min, x_max = data[:, 0].min() - 0.5, data[:, 0].max() + 0.5
    y_min, y_max = data[:, 1].min() - 0.5, data[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_tensor = torch.tensor(grid, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        Z = model(grid_tensor).numpy()
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, levels=[0, 0.5, 1], cmap=plt.cm.viridis, alpha=0.8)
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap=plt.cm.viridis)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title(f"Decision Boundary after Epoch {epoch+1}")
    plt.show()


def train_model(
    model: nn.Sequential,
    data_train: torch.Tensor,
    labels_train: torch.Tensor,
    data: np.ndarray,
    labels: np.ndarray,
    num_epochs: int = 10,
) -> None:
    loss_func = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.03)

    # Split data into chunks
    data_chunks = data_train.chunk(num_epochs)
    labels_chunks = labels_train.chunk(num_epochs)

    for epoch, (data_chunk, labels_chunk) in enumerate(zip(data_chunks, labels_chunks)):
        model.train()
        optimizer.zero_grad()
        output = model(data_chunk)
        loss = loss_func(output, labels_chunk.view(-1, 1))
        loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()*100:.2f}%")

        # Plot decision boundary after each epoch
        plot_decision_boundary(model, data, labels, epoch)


def main():
    # Data preparation
    N = 150
    train_pct = 0.8
    seed = 0x2024_06_25
    data_train, labels_train, data_test, labels_test, data_valid, labels_valid = (
        prepare_data(N, train_pct, seed)
    )

    # Plot data
    plot_data(data_train, labels_train)

    # Convert to torch tensors
    data_train_tensor = torch.tensor(data_train, dtype=torch.float32)
    labels_train_tensor = torch.tensor(labels_train, dtype=torch.float32)

    # Model creation
    model = create_model()

    # Model training with plotting decision boundary
    train_model(model, data_train_tensor, labels_train_tensor, data_train, labels_train)


if __name__ == "__main__":
    main()
