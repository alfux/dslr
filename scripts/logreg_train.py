import sys
import argparse
import traceback
from typing import Callable

import numpy as np
import pandas as pd


class SortingHatLogreg:
    """Computes the sorting hat coefficient by logistic regression."""

    def __init__(self: any, data: pd.DataFrame, epsilon: float = 1e-2):
        """Starts a logistic regression for each house."""
        self._epsilon = epsilon
        self._data = self._pre_process_data(data.drop(
            ["Divination", "History of Magic", "Transfiguration", "Flying"],
            axis=1))
        self._raven = self._descent(self._observed_ravenclaw)
        print("\rRavenclaw: complete !")
        self._data = self._pre_process_data(data.drop(
            ["Transfiguration", "Charms", "Flying", "Muggle Studies",
             "History of Magic"], axis=1))
        self._slyth = self._descent(self._observed_slytherin)
        print("\rSlytherin: complete !")
        self._data = self._pre_process_data(data.drop(
            ["Divination", "Muggle Studies", "Charms"], axis=1))
        self._gryff = self._descent(self._observed_gryffindor)
        print("\rGryffindor: complete !")
        self._data = self._pre_process_data(data.drop(
            ["Divination", "Muggle Studies", "History of Magic",
             "Transfiguration", "Flying"], axis=1))
        self._huffl = self._descent(self._observed_hufflepuff)
        print("\rHufflepuff: complete !")
        self.logreg_coefficients = pd.DataFrame(
            {"R": self._raven, "S": self._slyth,
             "G": self._gryff, "H": self._huffl})

    def _pre_process_data(self: any, data: pd.DataFrame) -> pd.DataFrame:
        """Pre-processes datas for logistic regression."""
        index = {k: i for (k, i) in zip(data.columns, range(data.shape[1]))}
        data = data.rename(index, axis=1)
        self._reduc_coef = [1] * data.shape[1]
        for i in range(1, data.shape[1]):
            (mean, reduc) = self._compute_means(data[i])
            data[i] = data[i].apply(lambda x: x / reduc if x == x else mean)
            self._reduc_coef[i] = reduc
        return data

    def _compute_means(self: any, data: pd.Series) -> tuple:
        """Computes data's mean and integer digits' mean."""
        no_nan_data = [x for x in data if x == x]
        (mean, reduc) = (0, 0)
        for i in range(len(no_nan_data)):
            reduc += self._count_digit(no_nan_data[i])
            mean += no_nan_data[i]
        reduc = 1 * 10 ** np.round(reduc / len(no_nan_data))
        mean = mean / (len(no_nan_data) * reduc)
        return (mean, reduc)

    def _count_digit(self: any, number: float) -> int:
        """Counts digits to the left of the dot."""
        i = 0
        number = np.abs(number)
        while number > 1:
            number /= 10
            i += 1
        return i

    def _descent(self: any, y: Callable) -> np.array:
        """Batch gradient descent to maximize log-likelihood."""
        self._y = y
        p = np.array([0] * len(self._data.columns))
        (nabla, norm) = self._gradient(p)
        while norm > self._epsilon:
            print(f"\r{np.clip(self._epsilon / norm, 0, 1):.2%}", end="")
            nabla /= norm
            p = p + self._learning_rate(p, nabla) * nabla
            (nabla, norm) = self._gradient(p)
        for i in range(len(p)):
            p[i] /= self._reduc_coef[i]
        return [p[i] if i < len(p) else 0 for i in range(14)]

    def _stochastic_descent(self: any, y: Callable) -> np.array:
        """Stochastic gradient descent to maximize log-likelihood."""
        self._y = y
        # p = np.array([0] * len(self._data.columns))

    def _learning_rate(self: any, p: np.array, nabla: np.array) -> float:
        """Computes an approximate of the optimal learning rate."""
        return (1e-0)

        def d_dt(t: float) -> float:
            """Derivative of line search log-likelihood."""
            d_dt = 0
            for i in range(self._data.shape[0]):
                xi = np.array(self._data.loc[i])
                yi = self._y(xi[0])
                xi[0] = 1
                yi_nabla_xi = yi * np.dot(nabla, xi)
                exposant = yi * np.dot(p, xi) + t * yi_nabla_xi
                d_dt += yi_nabla_xi / (1 + np.exp(exposant))
            return d_dt / self._data.shape[0]

        return self._bissection(d_dt)

    def _stochastic_learning_rate(self: any, y: Callable) -> float:
        """Computes an approximate of the optimal stochastic learning rate."""
        pass

    def _bissection(self: any, f: Callable) -> float:
        """Finds a zero from f by bissection method."""
        (a, b, step, fa, fb) = (-1, 1, 1, f(-1), f(1))
        while fa * fb > 0:
            a -= step
            b += step
            (fa, fb) = (f(a), f(b))
        while (np.abs(fa) > self._epsilon and
               np.abs(fb) > self._epsilon):
            c = (a + b) / 2
            fc = f(c)
            if np.sign(fa) * np.sign(fc) > 0:
                (a, fa) = (c, fc)
            else:
                (b, fb) = (c, fc)
        return a if np.abs(fa) < np.abs(fb) else b

    def _gradient(self: any, p: np.array) -> tuple[np.array, float]:
        """Computes the gradient of the log-likelihood in point p."""
        dim = self._data.shape[1]
        sum = np.array([0.0] * dim)
        for i in range(self._data.shape[0]):
            xi = [x for x in self._data.loc[i]]
            yi = self._y(xi[0])
            xi[0] = 1
            p_xi = np.dot(p, xi)
            sum += [yi * xi[j] / (1 + np.exp(yi * p_xi)) for j in range(dim)]
        sum /= self._data.shape[0]
        return (sum, np.linalg.norm(sum))

    def _stochastic_grad(self: any, p: np.array) -> tuple[np.array, float]:
        """Computes the gradient of the one-lined-log-likelihood in point p."""
        pass

    def _observed_ravenclaw(self: any, house: str) -> int:
        """Digital representation of the observed belonging to ravenclaw."""
        return 1 if house == "Ravenclaw" else -1

    def _observed_slytherin(self: any, house: str) -> int:
        """Digital representation of the observed belonging to slytherin."""
        return 1 if house == "Slytherin" else -1

    def _observed_gryffindor(self: any, house: str) -> int:
        """Digital representation of the observed belonging to gryffindor."""
        return 1 if house == "Gryffindor" else -1

    def _observed_hufflepuff(self: any, house: str) -> int:
        """Digital representation of the observed belonging to hufflepuff."""
        return 1 if house == "Hufflepuff" else -1


def main() -> None:
    """Trains a logistic regression model to mimic the Sorting Hat"""
    try:
        parser = argparse.ArgumentParser(sys.argv[0], f"{sys.argv[0]} [file]")
        parser.add_argument("file", help="csv file containing hogwarts data")
        data = pd.read_csv(parser.parse_args().file)
        data = data.drop(["Index", "First Name", "Last Name", "Arithmancy",
                          "Birthday", "Best Hand", "Astronomy", "Potions",
                          "Care of Magical Creatures"], axis="columns")
        SortingHat = SortingHatLogreg(data)
        SortingHat.logreg_coefficients.to_csv("../datasets/logreg_coefs.csv",
                                              index=False)
    except Exception as err:
        print(traceback.format_exc())
        print(f"{err.__class__.__name__}: {err}", sys.stderr)


if __name__ == "__main__":
    main()
