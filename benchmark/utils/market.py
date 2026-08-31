import os

import numpy
import pandas

MARKET_DIRECTORY = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "market_data")

DATASETS = ("factors_dataset", "sp500_dataset", "ftse100_dataset", "nasdaq_dataset")


def load_prices(dataset_index):
    path = os.path.join(MARKET_DIRECTORY, DATASETS[dataset_index] + ".csv.gz")
    return pandas.read_csv(path, index_col=0)


def returns_window(dataset_index, start_fraction, length, asset_cap):
    prices = load_prices(dataset_index)
    total = prices.shape[0]
    first = int(start_fraction * max(1, total - length))
    window = prices.iloc[first:first + length]
    window = window.dropna(axis=1, how="any")
    if window.shape[1] > asset_cap:
        window = window.iloc[:, :asset_cap]
    values = window.to_numpy(dtype=float)
    returns = values[1:] / values[:-1] - 1.0
    return numpy.ascontiguousarray(returns)
