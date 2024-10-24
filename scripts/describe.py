import sys
import argparse
import traceback
from typing import Generator

import pandas as pd
from pandas import DataFrame, Series

from tools import shorten, get_age


class Statistics:
    """Stores statistics in fields, based on the given datas"""
    field_names = ["Count", "Mean", "Var", "Unbiased Var", "Std",
                   "Less Biased Std", "Min", "10%", "20%", "25%", "30%", "40%",
                   "50%", "60%", "70%", "75%", "80%", "90%", "Max"]

    def __init__(self, data: Series) -> None:
        """Initializes each field with its processed value."""
        self.data = Statistics.quicksort([x for x in data if x == x])
        self.fields = {x: None for x in Statistics.field_names}
        self.fields["Count"] = len(self.data)
        self.fields["Min"] = self.data[0]
        self.fields["25%"] = self.percentile(25)
        self.fields["50%"] = self.percentile(50)
        self.fields["75%"] = self.percentile(75)
        self.fields["10%"] = self.percentile(10)
        self.fields["20%"] = self.percentile(20)
        self.fields["30%"] = self.percentile(30)
        self.fields["40%"] = self.percentile(40)
        self.fields["60%"] = self.percentile(60)
        self.fields["70%"] = self.percentile(70)
        self.fields["80%"] = self.percentile(80)
        self.fields["90%"] = self.percentile(90)
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
        self.fields["Unbiased Var"] = self.fields["Var"] * self.fields["Count"]
        self.fields["Unbiased Var"] /= self.fields["Count"] - 1
        self.fields["Std"] = self.fields["Var"] ** 0.5
        self.fields["Less Biased Std"] = self.fields["Unbiased Var"] ** 0.5

    def quicksort(data: list) -> list:
        """Quicksort sorting algorithm."""
        if Statistics.is_sorted(data):
            return data
        middle = len(data) // 2
        pivot = data[middle]
        data1 = [data[i] for i in range(len(data)) if data[i] <= pivot
                 and i != middle]
        data2 = [x for x in data if x > pivot]
        return (Statistics.quicksort(data1) + [pivot] +
                Statistics.quicksort(data2))

    def is_sorted(data: list) -> list:
        """Check if the list is sorted."""
        if len(data) < 2:
            return True
        for i in range(len(data) - 1):
            if data[i] > data[i + 1]:
                return False
        return True


def main() -> None:
    """Displays statistics on each courses datas from the argument."""
    try:
        parser = argparse.ArgumentParser(usage="describe.py [file] [-w width]")
        parser.add_argument("data", help="dataset to read")
        parser.add_argument("-w", help="max name size", default=15, type=int)
        av = parser.parse_args()
        data = pd.read_csv(av.data)
        data["Birthday"] = data["Birthday"].apply(lambda x: get_age(x))
        data = data.rename(columns={"Birthday": "Age"}).select_dtypes(float)
        if "Hogwarts House" in data.columns:
            data.drop("Hogwarts House", axis=1, inplace = True)
        pd.options.display.float_format = lambda x: f"{x:1g}"
        def statistics_generator() -> Generator:
            """Generator of Series of Statistics for each columns in data."""
            for x in data.columns:
                yield Series(Statistics(data[x]).fields, name=shorten(x, av.w))

        print(DataFrame(statistics_generator()).transpose())
    except Exception as err:
        print(f"{err.__class__.__name__}: {err}", file=sys.stderr)
        print(traceback.format_exc())


if __name__ == "__main__":
    main()
