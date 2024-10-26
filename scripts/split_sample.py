import sys
import os
import argparse as arg
from typing import Generator

import pandas as pd


def split_sample(sample: pd.DataFrame, n: int) -> list[pd.DataFrame]:
    """Split a sample into n sub samples."""
    sample = sample.sample(frac=1).reset_index(drop=True)
    size = sample.shape[0] // n

    def sample_generator() -> Generator[pd.DataFrame]:
        """Size n sample generator."""
        for i in range(n):
            yield sample[i * size:(i + 1) * size].reset_index(drop=True)
        if n * size < sample.shape[0]:
            yield sample[n * size:sample.shape[0]].reset_index(drop=True)

    return list(sample_generator())


def main() -> None:
    """Shuffles then splits dataset in two 20% - 80% files."""
    try:
        parser = arg.ArgumentParser(sys.argv[0], f"{sys.argv[0]} [file] [n]")
        parser.add_argument("file", help="file to split")
        parser.add_argument("number", help="number of splits")
        data = pd.read_csv(parser.parse_args().file).drop("Index", axis=1)
        samples = split_sample(data, int(parser.parse_args().number))
        try:
            os.mkdir("sub")
        except FileExistsError:
            pass
        i = 0
        for sub in samples:
            sub.index.name = "Index"
            sub.to_csv(f"sub/sample{i}.csv")
            i += 1
    except Exception as err:
        print(f"{err.__class__.__name__}: {err}", file=sys.stderr)


if __name__ == "__main__":
    main()
