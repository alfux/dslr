import argparse
import sys

from typing import Callable

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backend_bases import Event
from tools import course_gen, button_gen, close_callback
from tools import Figure, Button


def display_course(fig: Figure, data: tuple, color: dict) -> None:
    """Refresh the histogram."""
    ax = fig.get_axes()[-1]
    ax.clear()
    ax.set_title(data[0])
    for (sub_key, values) in data[1].items():
        ax.hist(values, color=color[sub_key], label=sub_key, bins=50)
    fig.legend(fancybox=True, shadow=True)
    fig.canvas.draw()


def main() -> None:
    """Display histograms of every features by houses."""
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("data", help="Raw dataset to read.")
        raw_data = pd.read_csv(parser.parse_args().data)
        data = raw_data.select_dtypes(float)
        course = list(course_gen(raw_data, data))
        color = {"Ravenclaw": (0.3, 0, 1, 1), "Slytherin": (0, 0.5, 0, 1),
                 "Gryffindor": (1, 0, 0, 1), "Hufflepuff": (1, 0.7, 0, 1)}
        fig = plt.figure("Histograms", facecolor=(0.8, 0.8, 0.8))
        fig.set_size_inches(15, 9)

        def create_callback(i: int) -> Callable[[Event], None]:
            """Creates a callback function relative to i's value."""
            return lambda event: display_course(fig, course[i], color)

        buttons: list[Button] = list(button_gen(fig, data.columns))
        for i in range(len(buttons)):
            buttons[i].on_clicked(create_callback(i))
        fig.add_subplot(position=(0.125, 0.15, 0.75, 0.75))
        display_course(fig, course[10], color)
        fig.canvas.mpl_connect("key_release_event", close_callback)
        plt.show()
    except Exception as err:
        print(f"{err.__class__.__name__}: {err}", file=sys.stderr)


if __name__ == "__main__":
    main()
