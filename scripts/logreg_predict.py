import argparse
import sys
import traceback

import pandas as pd
from pandas import DataFrame, Series
import numpy as np


def pre_process_data(data: DataFrame, weights: DataFrame) -> tuple[list, list]:
    """Pre-processes datas for logistic computations."""
    for i in range(1, data.shape[1]):
        data[data.columns[i]] = data[data.columns[i]].apply(
            lambda x: x if x == x else data_mean(data[data.columns[i]]))
        raven = reindex(data.drop(["Divination", "History of Magic",
                                   "Transfiguration", "Flying"], axis=1))
        wr = [weights.at[i, 'R'] for i in range(len(raven.columns))]
        slyth = reindex(data.drop(["Transfiguration", "Charms", "Flying",
                                   "Muggle Studies", "History of Magic"],
                                  axis=1))
        ws = [weights.at[i, 'S'] for i in range(len(slyth.columns))]
        gryff = reindex(data.drop(["Divination", "Muggle Studies", "Charms"],
                                  axis=1))
        wg = [weights.at[i, 'G'] for i in range(len(gryff.columns))]
        huffl = reindex(data.drop(["Divination", "Muggle Studies",
                                   "History of Magic", "Transfiguration",
                                   "Flying"], axis=1))
        wh = [weights.at[i, 'H'] for i in range(len(huffl.columns))]
    return ([raven, slyth, gryff, huffl], [wr, ws, wg, wh])


def reindex(data: DataFrame) -> DataFrame:
    """Reindexes datas with number."""
    index = {k: i for (k, i) in zip(data.columns, range(data.shape[1]))}
    data = data.rename(index, axis=1)
    return data


def data_mean(data: np.array) -> float:
    """Computes mean of data avoiding NaNs"""
    data = [x for x in data if x == x]
    """Choose a house for a stud based on results."""
    return np.sum(data) / len(data)


def sorting_hat(i: int, stud: list[Series], weights: list[DataFrame]) -> int:
    stud = [stud[0].loc[i], stud[1].loc[i], stud[2].loc[i], stud[3].loc[i]]
    expected = stud[0][0]
    for i in range(4):
        stud[i] = stud[i].drop([0])
    odds = [logistic(stud[i], weights[i]) for i in range(4)]
    houses = ["Ravenclaw", "Slytherin", "Gryffindor", "Hufflepuff"]
    predicted = houses[odds.index(np.max(odds))]
    print(f"expected: {expected} -> {predicted}")
    return 1 if predicted == expected else 0


def logistic(x: Series, weights: Series) -> float:
    """Computes the logistic function's value in x according to wieghts."""
    return 1 / (1 + np.exp(-np.dot([1, *x], weights)))


def main() -> None:
    """Use trained weights to make predictions on a dataset."""
    try:
        parser = argparse.ArgumentParser(
            sys.argv[0], f"{sys.argv[0]} [dataset] [weights]")
        parser.add_argument("dataset", help="dataset used for predictions")
        parser.add_argument("weights", help="weights from logreg training")
        weights = pd.read_csv(parser.parse_args().weights)
        data = pd.read_csv(parser.parse_args().dataset)
        data = data.drop(["Index", "First Name", "Last Name", "Arithmancy",
                          "Birthday", "Best Hand", "Astronomy", "Potions",
                          "Care of Magical Creatures"], axis="columns")
        (stud, weights) = pre_process_data(data, weights)
        precision = 0
        for i in range(data.shape[0]):
            precision += sorting_hat(i, stud, weights)
        print(f"The sorting hat was {precision / data.shape[0]:.2%} right")
    except Exception as err:
        print(traceback.format_exc())
        print(f"{err.__class__.__name__}: {err}", sys.stderr)


if __name__ == "__main__":
    main()
