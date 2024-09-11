import argparse
import pandas
import sys


def quicksort(v: list) -> list:
    """Quicksort algorithm to sort a list"""
    n = len(v)

    def is_sorted() -> None:
        if n < 2:
            return (True)
        prev = v[0]
        for x in v:
            if prev > x:
                return (False)
            else:
                prev = x
        return (True)

    if is_sorted():
        return (v)
    v1 = list()
    v2 = list()
    for i in range(n - 1):
        if (v[i] < v[-1]):
            v1 += [v[i]]
        else:
            v2 += [v[i]]
    return (quicksort(v1) + [v[-1]] + quicksort(v2))


def get_stats(data: list) -> list:
    """Generator for stats of each data columns."""
    data = [x for x in data if x == x]
    data = quicksort(data)
    n = len(data)

    def per(p: int) -> float | int:
        p = (p * (n - 1) / 100)
        r = p % 1
        p = int(p - r)
        return (data[p] + r * (data[p + 1] - data[p]))

    def mean_std():
        mean = 0
        std = 0
        for x in data:
            mean += x
        mean /= n
        std = 0
        for x in data:
            std += (mean - x) ** 2
        std = (std / n) ** 0.5
        return (mean, std)

    return ([n, *mean_std(), data[0], per(25), per(50), per(75), data[-1]])


def main() -> None:
    """Entrypoint of the application."""
    try:
        parser = argparse.ArgumentParser(usage="Describes the data file.")
        parser.add_argument("data", help="Dataset to be read.")
        data = pandas.read_csv(parser.parse_args().data)
        data = data.select_dtypes(include=(float))
        stat = pandas.DataFrame(
            {x: get_stats(data[x].values) for x in data.columns},
            ["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"])
        print(stat)
        return (None)
    except Exception as err:
        print(f"{err.__class__.__name__}: {err}", file=sys.stderr)


if __name__ == "__main__":
    main()
