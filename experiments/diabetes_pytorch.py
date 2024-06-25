import numpy as np
import polars as pl
import torch
from sklearn.datasets import load_diabetes


def get_data() -> tuple[pl.DataFrame, pl.Series]:
    dataset = load_diabetes()
    data = pl.DataFrame(
        dataset["data"], schema={col: pl.Float32 for col in dataset["feature_names"]}
    )
    target = pl.Series("target", dataset["target"], dtype=pl.Float32)
    return data, target


def prepare_data(training_pct: float = 0.8) -> tuple[pl.DataFrame, pl.Series]:
    data, target = get_data()
    assert len(data) == len(target)

    N = len(data)
    train_n = int(N * training_pct)
    test_n = (N - train_n) // 2
    eval_n = N - train_n - test_n
    assert train_n + test_n + eval_n == N

    data_train = data.slice(0, train_n)
    target_train = target.slice(0, train_n)
    data_test = data.slice(train_n, train_n + test_n)
    target_test = target.slice(train_n, train_n + test_n)
    data_eval = data.slice(train_n + test_n, N)
    target_eval = target.slice(train_n + test_n, N)
    return data_train, target_train, data_test, target_test, data_eval, target_eval


if __name__ == "__main__":
    data_train, target_train, data_test, target_test, data_eval, target_eval = (
        prepare_data()
    )
    print(type(data_train))
    print(type(target_train))
    print(type(data_test))
    print(type(target_test))
    print(type(data_eval))
    print(type(target_eval))
    print("Done")
