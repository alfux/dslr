import sys
import argparse
from typing import Generator

import pandas as pd
from pandas import DataFrame, Series


class Statistics:
    """Stores statistics in fields, based on the given datas"""
    field_names = ["Count", "Mean", "Var", "Std", "Min", "25%", "50%", "75%",
                   "Max"]

    def __init__(self, data: Series) -> None:
        """Initializes each field with its processed value."""
        self.data = Statistics.quicksort([x for x in data if x == x])
        self.fields = {x: None for x in Statistics.field_names}
        self.fields["Count"] = len(self.data)
        self.fields["Min"] = self.data[0]
        self.fields["25%"] = self.percentile(25)
        self.fields["50%"] = self.percentile(50)
        self.fields["75%"] = self.percentile(75)
        self.fields["Max"] = self.data[-1]
        self.compute_mean_var_std()

    def percentile(self, p: int) -> float | int:
        """Compute sorted list p'th percentile."""
        p = (p * (self.fields["Count"] - 1) / 100)
        r = p % 1
        p = int(p - r)
        return (self.data[p] + r * (self.data[p + 1] - self.data[p]))

    def compute_mean_var_std(self) -> None:
        """Compute mean, variance and standard deviation with a single read."""
        self.fields["Mean"] = 0
        self.fields["Var"] = 0
        for x in self.data:
            self.fields["Mean"] += x
            self.fields["Var"] += x ** 2
        self.fields["Mean"] /= self.fields["Count"]
        self.fields["Var"] /= self.fields["Count"]
        self.fields["Var"] -= self.fields["Mean"] ** 2
        self.fields["Std"] = self.fields["Var"] ** 0.5

    def quicksort(data: list) -> list:
        """Quicksort sorting algorithm."""
        if len(data) < 2:
            return data
        pivot = data[len(data) // 2]
        data1 = [x for x in data if x < pivot]
        data2 = [x for x in data if x > pivot]
        return (Statistics.quicksort(data1) + [pivot] +
                Statistics.quicksort(data2))


def main() -> None:
    """Displays statistics on each courses datas from the argument."""
    try:
        parser = argparse.ArgumentParser(usage="Describes the data file.")
        parser.add_argument("data", help="Dataset to be read.")
        data = pd.read_csv(parser.parse_args().data).select_dtypes(float)

        def statistics_generator() -> Generator:
            """Generator of Series of Statistics for each columns in data."""
            for x in data.columns:
                yield Series(Statistics(data[x]).fields, name=x)

        print(DataFrame(statistics_generator()))
    except Exception as err:
        print(f"{err.__class__.__name__}: {err}", file=sys.stderr)


if __name__ == "__main__":
    main()
