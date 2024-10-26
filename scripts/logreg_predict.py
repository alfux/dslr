import argparse as arg
import sys
from typing import Generator, Self

import pandas as pd
from pandas import DataFrame, Series
import numpy as np


class SortingHat:
    """Hogwarts Sorting Hat."""

    def __init__(self: Self, data: DataFrame, weights: DataFrame) -> None:
        """Sort students from data according to weights."""
        data = data.drop(["Index", "First Name", "Last Name", "Arithmancy",
                          "Birthday", "Best Hand", "Astronomy", "Potions",
                          "Care of Magical Creatures"], axis="columns")
        (self._stud, self._weights) = self._pre_process(data, weights)
        self.result = DataFrame(self._sorting_hat(),
                                columns=["Hogwarts House"])
        self.result.index.name = "Index"

    def _pre_process(self: Self, data: DataFrame, weights: DataFrame) -> tuple:
        """Pre-processes datas for logistic computations."""
        data = data.dropna(subset=data.columns.drop("Hogwarts House"))
        data.reset_index(inplace=True, drop=True)
        raven = self._reindex(data.drop(["Divination", "History of Magic",
                                         "Transfiguration", "Flying"], axis=1))
        slyth = self._reindex(data.drop(["Transfiguration", "Charms", "Flying",
                                         "Muggle Studies", "History of Magic"],
                                        axis=1))
        gryff = self._reindex(data.drop(["Divination", "Muggle Studies",
                                         "Charms"], axis=1))
        huffl = self._reindex(data.drop(
            ["Divination", "Muggle Studies", "Flying", "History of Magic",
             "Transfiguration"], axis=1))
        for i in range(1, data.shape[1]):
            wr = [weights.at[i, 'R'] for i in range(len(raven.columns))]
            ws = [weights.at[i, 'S'] for i in range(len(slyth.columns))]
            wg = [weights.at[i, 'G'] for i in range(len(gryff.columns))]
            wh = [weights.at[i, 'H'] for i in range(len(huffl.columns))]
        return ([raven, slyth, gryff, huffl], [wr, ws, wg, wh])

    def _reindex(self: Self, data: DataFrame) -> DataFrame:
        """Reindexes datas with number."""
        index = {k: i for (k, i) in zip(data.columns, range(data.shape[1]))}
        data = data.rename(index, axis=1)
        return data

    def _sorting_hat(self: Self) -> Generator:
        """Choose a house for a stud based on results."""
        self.precision = 0
        total = 0
        houses = ["Ravenclaw", "Slytherin", "Gryffindor", "Hufflepuff"]
        for i in range(self._stud[0].shape[0]):
            line = [self._stud[0].loc[i], self._stud[1].loc[i],
                    self._stud[2].loc[i], self._stud[3].loc[i]]
            expected = line[0][0]
            total += 1 if expected == expected else 0
            for i in range(4):
                line[i] = line[i].drop([0])
            odds = [self._logistic(line[i], i) for i in range(4)]
            predicted = houses[odds.index(np.max(odds))]
            self.precision += 1 if predicted == expected else 0
            yield predicted
        self.precision = self.precision / total if total != 0 else None

    def _logistic(self: Self, x: Series, i: int) -> float:
        """Computes the logistic function's value in x according to wieghts."""
        return 1 / (1 + np.exp(-np.dot([1, *x], self._weights[i])))


def main() -> None:
    """Use trained weights to make predictions on a dataset."""
    try:
        parser = arg.ArgumentParser(
            sys.argv[0], f"{sys.argv[0]} [dataset] [weights]")
        parser.add_argument("dataset", help="dataset used for predictions")
        parser.add_argument("weights", help="weights from logreg training")
        data = pd.read_csv(parser.parse_args().dataset)
        weights = pd.read_csv(parser.parse_args().weights)
        sorting_hat = SortingHat(data, weights)
        sorting_hat.result.to_csv("houses.csv")
        if sorting_hat.precision is not None:
            print(f"The sorting hat was {sorting_hat.precision:.2%} right")
    except Exception as err:
        print(f"{err.__class__.__name__}: {err}", file=sys.stderr)


if __name__ == "__main__":
    main()
