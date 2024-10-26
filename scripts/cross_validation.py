import argparse as arg
import sys
import warnings

import numpy as np
import pandas as pd
from pandas import DataFrame

from logreg_train import SortingHatLogreg
from logreg_predict import SortingHat
from split_sample import split_sample


def compute_precision(samples: list[DataFrame], **kwargs: dict) -> float:
    """Iterates over samples to compute mean precision."""
    precision = 0
    for i in range(len(samples)):
        print(f"----------Iteration-{i}----------", end="\n\n")
        training_set = [samples[j] for j in range(len(samples)) if j != i]
        if len(training_set) > 0:
            training_set = pd.concat(training_set)
            training_set.reset_index(inplace=True, drop=True)
        else:
            training_set = samples[i]
        predict_set = samples[i]
        training = SortingHatLogreg(training_set, **kwargs)
        predict = SortingHat(predict_set, training.logreg_coef)
        precision += predict.precision
        print(f"Iteration's precision: {predict.precision:.2%}", end="\n\n")
    return precision / len(samples)


def main() -> None:
    """Trains over sub-samples to compute precision avoiding over(under)fit."""
    try:
        warnings.filterwarnings(action="ignore")
        parser = arg.ArgumentParser(
            sys.argv[0], f"{sys.argv[0]} [file] [N] [-ns] [-m size] [-e prec]")
        parser.add_argument("file", help="csv file containing hogwarts datas")
        parser.add_argument("N", help="size of the sub-samples")
        parser.add_argument("-s", "--stochastic-gd", action="store_true",
                            help="use stochastic gradient descent", )
        parser.add_argument("-m", "--mini-batch-gd",
                            help="use mini-batch gradient descent")
        parser.add_argument("-e", "--epsilon", help="sets epsilon precision")
        parser.add_argument("-n", "--newton-raphson", action="store_true",
                            help="use newton-raphson algorithm")
        samples = split_sample(pd.read_csv(parser.parse_args().file),
                               np.abs(int(parser.parse_args().N)))
        kwargs = {"epsilon": parser.parse_args().epsilon,
                  "batch": parser.parse_args().mini_batch_gd,
                  "sgd": parser.parse_args().stochastic_gd,
                  "nr": parser.parse_args().newton_raphson}
        precision = compute_precision(samples, **kwargs)
        print(f"Overall model's precision is {precision:.2%}")
    except Exception as err:
        print(f"{err.__class__.__name__}: {err}", file=sys.stderr)


if __name__ == "__main__":
    main()
