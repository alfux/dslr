import argparse
import sys
import warnings

import pandas as pd
import numpy as np


def pre_process_data(data: pd.DataFrame) -> pd.DataFrame:
    """Pre-processes datas for logistic computations."""
    index = {k: i for (k, i) in zip(data.columns, range(data.shape[1]))}
    data = data.rename(index, axis=1)
    for i in range(1, data.shape[1]):
        data[i] = data[i].apply(lambda x: x if x == x else data_mean(data[i]))
    return data


def data_mean(data: np.array) -> float:
    """Computes mean of data avoiding NaNs"""
    data = [x for x in data if x == x]
    return np.sum(data) / len(data)


def sorting_hat(student: pd.Series, weights: pd.Series) -> None:
    """Choose a house for a student based on results."""
    expected = student[0]
    student = student.drop(0)
    odds = [logistic(student, weights["R"]), logistic(student, weights["S"]),
            logistic(student, weights["G"]), logistic(student, weights["H"])]
    houses = ["Ravenclaw", "Slytherin", "Gryffindor", "Hufflepuff"]
    print(f"expected: {expected} -> {houses[odds.index(np.max(odds))]}")


def logistic(x: pd.Series, weights: pd.Series) -> float:
    """Computes the logistic function's value in x according to wieghts."""
    return 1 / (1 + np.exp(-np.dot([1, *x], weights)))


def main() -> None:
    """Use trained weights to make predictions on a dataset."""
    try:
        warnings.simplefilter("ignore")
        parser = argparse.ArgumentParser(
            sys.argv[0], f"{sys.argv[0]} [dataset] [weights]")
        parser.add_argument("dataset", help="dataset used for predictions")
        parser.add_argument("weights", help="weights from logreg training")
        weights = pd.read_csv(parser.parse_args().weights)
        data = pd.read_csv(parser.parse_args().dataset)
        data = data.drop(["Index", "First Name", "Last Name",
                          "Birthday", "Best Hand"], axis="columns")
        data = pre_process_data(data)
        for i in range(data.shape[0]):
            sorting_hat(data.loc[i], weights)
    except Exception as err:
        print(f"{err.__class__.__name__}: {err}", sys.stderr)


if __name__ == "__main__":
    main()
