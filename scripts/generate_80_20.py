import sys
import argparse

import pandas as pd


def main() -> None:
    """Shuffles then splits dataset in two 20% - 80% files."""
    try:
        parser = argparse.ArgumentParser(sys.argv[0], f"{sys.argv[0]} [file]")
        parser.add_argument("file", help="file to split")
        data = pd.read_csv(parser.parse_args().file).drop("Index", axis=1)
        data = data.sample(frac=1)
        data.reset_index(inplace=True, drop=True)
        size = 20 * data.shape[0] // 100
        testing_sample = data[0 : size].reset_index(drop=True)
        testing_sample.index.name = "Index"
        testing_sample.to_csv("testing_sample.csv")
        training_samlpe = data[size : data.shape[0]].reset_index(drop=True)
        training_samlpe.index.name = "Index"
        training_samlpe.to_csv("training_sample.csv")
    except Exception as err:
        print(f"main(): {err.__class__.__name__}: {err}")


if __name__ == "__main__":
    main()