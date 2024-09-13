import argparse
import sys

import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.backend_bases import KeyEvent

from tools import courses_gen, shorten


class PairPlot:
    """Manages the pair plot."""
    def __init__(self, name: str, raw_data: DataFrame) -> None:
        """Sets datas to create a scatter plot matrix based on raw_data."""
        self.fig = plt.figure(name)
        self.fig.set_facecolor((.8, .8, .8))
        self.fig.subplots_adjust(.05, 0.02, .99, .93)
        self.fig.set_size_inches(17, 9.5)
        self.fig.canvas.mpl_connect("key_press_event", PairPlot.close_callback)
        self.raw_data = raw_data
        self.data = raw_data.select_dtypes(float)
        self.courses = list(courses_gen(self.raw_data, self.data))
        self.houses = self.courses[0][1].keys()
        self.colors = {"Ravenclaw": (.3, 0, 1, 1), "Slytherin": (0, .5, 0, 1),
                       "Gryffindor": (1, 0, 0, 1), "Hufflepuff": (1, .7, 0, 1)}
        self.set_subplots()
        self.fig.legend(self.houses, fancybox=True, shadow=True, ncol=4,
                        loc="upper center")

    def close_callback(event: KeyEvent) -> None:
        """Callback function to close the app with escape key."""
        if event.key == "escape":
            plt.close()

    def set_subplots(self) -> None:
        """Sets every subplots datas."""
        dim = len(self.data.columns)
        for i in range(dim):
            for j in range(dim):
                axes = self.fig.add_subplot(dim, dim, i * dim + j + 1)
                self.set_axes(axes, i, j)

    def set_axes(self, axes: Axes, i: int, j: int) -> None:
        """Sets axes datas of a single subplot."""
        for h in self.houses:
            if i == j:
                axes.hist(self.courses[i][1][h], bins=50, color=self.colors[h],
                          label=h)
            else:
                axes.scatter(self.courses[j][1][h], self.courses[i][1][h],
                             color=self.colors[h], label=h, sizes=[0.1])
        if j > 0:
            axes.tick_params('y', labelleft=False)
        else:
            axes.tick_params('y', labelsize=5)
            axes.set_ylabel(shorten(self.courses[i][0], 10), {"fontsize": 7})
            axes.yaxis.set_label_coords(-.5, .5)
        if i == 0:
            axes.tick_params('x', labeltop=True, labelsize=5)
            axes.set_xlabel(shorten(self.courses[j][0]), {"fontsize": 7})
            axes.xaxis.set_label_position("top")
        else:
            axes.tick_params('x', labelbottom=False)

    def update(self):
        """Update the pair plot figure."""
        self.fig.canvas.draw()


def main() -> None:
    """Displays a scatter plot matrix of the datas."""
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("raw_data", help="Raw dataset to read.")
        raw_data = pd.read_csv(parser.parse_args().raw_data)
        pair_plot = PairPlot("Pair plot", raw_data)
        pair_plot.update()
        plt.show()
    except Exception as err:
        print(f"{err.__class__.__name__}: {err}", file=sys.stderr)


if __name__ == "__main__":
    main()
