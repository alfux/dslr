import sys
import argparse
from typing import Self, Callable, Generator

import numpy as np
import pandas as pd


class SortingHatLogreg:
    """Computes the sorting hat coefficient by logistic regression."""

    def __init__(self: Self, data: pd.DataFrame,
                 descent_epsilon: float = 1e-4,
                 bissection_epsilon: float = 1e-10):
        """Starts a logistic regression for each house."""
        self._data = self._pre_process_data(data)
        self._bissection_epsilon = bissection_epsilon
        self._descent_epsilon = descent_epsilon
        self._raven = self._descent(self._observed_ravenclaw)
        self._slyth = self._descent(self._observed_slytherin)
        self._gryff = self._descent(self._observed_gryffindor)
        self._huffl = self._descent(self._observed_hufflepuff)
        self.logreg_coefficients = pd.DataFrame(
            {"R": self._raven, "S": self._slyth,
             "G": self._gryff, "H": self._huffl})

    def _pre_process_data(self: Self, data: pd.DataFrame) -> pd.DataFrame:
        """Pre-processes datas for logistic regression."""
        index = {k: i for (k, i) in zip(data.columns, range(data.shape[1]))}
        data = data.rename(index, axis=1)
        self._reduc_coef = [14] * data.shape[1]
        for i in range(1, data.shape[1]):
            (mean, reduc) = self._compute_means(data[i])
            data[i] = data[i].apply(lambda x: x / reduc if x == x else mean)
            self._reduc_coef[i] = reduc
        return data

    def _compute_means(self: Self, data: pd.Series) -> tuple:
        """Computes data's mean and integer digits' mean."""
        no_nan_data = [x for x in data if x == x]
        (mean, reduc) = (0, 0)
        for i in range(len(no_nan_data)):
            reduc += self._count_digit(no_nan_data[i])
            mean += no_nan_data[i]
        reduc = 14 * 10 ** np.round(reduc / len(no_nan_data))
        mean = mean / (len(no_nan_data) * reduc)
        return (mean, reduc)

    def _count_digit(self: Self, number: float) -> int:
        """Counts digits to the left of the dot."""
        i = 0
        number = np.abs(number)
        while number > 1:
            number /= 10
            i += 1
        return i

    def _descent(self: Self, y: Callable) -> np.array:
        """Returns approximation of point p where log-likelihood is maximal."""
        self._y = y
        p = np.array([0] * len(self._data.columns))
        nabla = self._gradient(p)
        norm = np.linalg.norm(nabla)
        self._iteration = 1
        print(norm)
        while norm > self._descent_epsilon:
            nabla /= norm
            p = p + self._learning_rate(p, nabla) * nabla
            nabla = self._gradient(p)
            norm = np.linalg.norm(nabla)
            self._iteration += 1
            print(f"{self._iteration}: {np.linalg.norm(nabla)}")
        for i in range(0, len(p)):
            p[i] /= self._reduc_coef[i]
        return p

    def _learning_rate(self: Self, p: np.array, nabla: np.array) -> float:
        """Computes an approximate of the optimal learning rate."""

        def d_dt(t: float) -> float:
            """(first, second) derivative of line search log-likelihood."""
            d_dt = 0
            for i in range(self._data.shape[0]):
                xi = np.array(self._data.loc[i])
                yi = self._y(xi[0])
                xi[0] = 1 / 14
                yi_nabla_xi = yi * np.dot(nabla, xi)
                exposant = yi * np.dot(p, xi) + t * yi_nabla_xi
                if (exposant > 700):
                    print(i, xi)
                    quit()
                d_dt += yi_nabla_xi / (1 + np.exp(exposant))
            return d_dt / self._data.shape[0]

        return self._bissection(d_dt)

    def _bissection(self: Self, f: Callable) -> float:
        """Finds a zero from f by bissection method."""
        (a, b, step, fa, fb) = (-1, 1, 1, f(-1), f(1))
        while fa * fb > 0:
            a -= step
            b += step
            (fa, fb) = (f(a), f(b))
        while (np.abs(fa) > self._bissection_epsilon and
               np.abs(fb) > self._bissection_epsilon):
            c = (a + b) / 2
            fc = f(c)
            if np.sign(fa) * np.sign(fc) > 0:
                (a, fa) = (c, fc)
            else:
                (b, fb) = (c, fc)
        return a if np.abs(fa) < np.abs(fb) else b

    def _gradient(self: Self, p: np.array) -> np.array:
        """Computes the gradient of the log-likelihood in point p."""

        def partial_derivative_generator() -> Generator:
            """Computes partial derivative in the jth direction."""
            for j in range(self._data.shape[1]):
                sum = 0
                for i in range(self._data.shape[0]):
                    xi = [x for x in self._data.loc[i]]
                    yi = self._y(xi[0])
                    xi[0] = 1 / 14
                    sum += yi * xi[j] / (1 + np.exp(yi * np.dot(p, xi)))
                yield sum / self._data.shape[0]

        return np.fromiter(partial_derivative_generator(), float)

    def _observed_ravenclaw(self: Self, house: str) -> int:
        """Digital representation of the observed belonging to ravenclaw."""
        return 1 if house == "Ravenclaw" else -1

    def _observed_slytherin(self: Self, house: str) -> int:
        """Digital representation of the observed belonging to slytherin."""
        return 1 if house == "Slytherin" else -1

    def _observed_gryffindor(self: Self, house: str) -> int:
        """Digital representation of the observed belonging to gryffindor."""
        return 1 if house == "Gryffindor" else -1

    def _observed_hufflepuff(self: Self, house: str) -> int:
        """Digital representation of the observed belonging to hufflepuff."""
        return 1 if house == "Hufflepuff" else -1

    def logistic(p: list, x: list):
        """Computes the logistic function of parameter p in point x."""
        x = [n if n == n else 0 for n in x]
        return 1 / (1 + np.exp(-np.dot(p, np.array(x))))


def main() -> None:
    """Trains a logistic regression model to mimic the Sorting Hat"""
    try:
        parser = argparse.ArgumentParser(sys.argv[0], f"{sys.argv[0]} [file]")
        parser.add_argument("file", help="csv file containing hogwarts data")
        data = pd.read_csv(parser.parse_args().file)
        data = data.drop(["Index", "First Name", "Last Name", "Birthday",
                          "Best Hand"], axis="columns")
        SortingHat = SortingHatLogreg(data)
        SortingHat.logreg_coefficients.to_csv("./datasets/logreg_coefs.csv",
                                              index=False)
    except Exception as err:
        print(f"{err.__class__.__name__}: {err}", sys.stderr)


if __name__ == "__main__":
    main()
