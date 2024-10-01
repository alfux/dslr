import sys
import argparse

import numpy as np
import pandas as pd

def main() -> None:
    """Trains a logistic regression model to mimic the Sorting Hat"""
    try:
        parser = argparse.ArgumentParser(sys.argv[0], "logreg_train.py [file]")
        parser.add_argument("file", help="csv file containing hogwarts data")
        raw_data = pd.read_csv(parser.parse_args().file)
        # start logistic regression here
    except Exception as err:
        print(f"{err.__class__.__name__}: {err}")


if __name__ == "__main__":
    main()