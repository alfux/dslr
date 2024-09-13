import argparse
import sys
from typing import Callable

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseEvent, MouseButton

from tools import courses_gen, button_gen, close_callback
from tools import Figure, Button


def display_selected(fig: Figure, selected: list, color: dict) -> None:
    """Displays selected data on the scatter plot."""
    axis = fig.get_axes()[-1]
    axis.clear()
    for key in selected[0][0][1].keys():
        axis.scatter(selected[0][0][1][key], selected[1][0][1][key],
                     label=key, color=color[key])
    axis.set_title(f"{selected[1][0][0]} / {selected[0][0][0]}")
    axis.set_xlabel(selected[0][0][0])
    axis.set_ylabel(selected[1][0][0])
    fig.legend(fancybox=True, shadow=True)
    fig.canvas.draw()


def button_callback(i: int, fig: Figure, selected: tuple, buttons: list,
                    bcolor: list, courses: list, color: dict) -> Callable:
    """Callback function to manage plot when clicking buttons."""
    def display_selected_callback(event: MouseEvent) -> None:
        side = 0 if event.button == MouseButton.RIGHT else 1
        other = 1 - side
        if selected[0][1] != selected[1][1]:
            buttons[selected[side][1]].ax.set_facecolor(bcolor[2])
            buttons[selected[side][1]].color = bcolor[2]
        else:
            buttons[selected[other][1]].ax.set_facecolor(bcolor[other])
            buttons[selected[other][1]].color = bcolor[other]
        buttons[i].ax.set_facecolor(bcolor[side])
        buttons[i].color = bcolor[side]
        selected[side] = (courses[i], i)
        display_selected(fig, selected, color)
    return display_selected_callback


def init_selected_buttons(buttons: list[Button], bcolor: list[tuple]):
    """Sets color for starting selected buttons."""
    buttons[3].ax.set_facecolor(bcolor[0])
    buttons[3].color = bcolor[0]
    buttons[1].ax.set_facecolor(bcolor[1])
    buttons[1].color = bcolor[1]


def main() -> None:
    """Displays a scatter plot comparing courses."""
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("raw_data", help="Raw dataset to read.")
        raw_data = pd.read_csv(parser.parse_args().raw_data)
        data = raw_data.select_dtypes(float)
        fig = plt.figure("Scatter plots", facecolor=(0.8, 0.8, 0.8))
        fig.set_size_inches(15, 9)
        fig.canvas.mpl_connect("key_release_event", close_callback)
        courses = list(courses_gen(raw_data, data))
        color = {"Ravenclaw": (0.3, 0, 1, 1), "Slytherin": (0, 0.5, 0, 1),
                 "Gryffindor": (1, 0, 0, 1), "Hufflepuff": (1, 0.7, 0, 1)}
        bcolor = ((0, 1, 1, 1), (1, 0, 1, 1), (1, 1, 1))
        buttons = list[Button](button_gen(fig, data.columns))
        selected = [(courses[3], 3), (courses[1], 1)]
        for i in range(len(buttons)):
            buttons[i].on_clicked(button_callback(
                i, fig, selected, buttons, bcolor, courses, color))
        fig.add_subplot(position=(0.125, 0.15, 0.75, 0.75))
        init_selected_buttons(buttons, bcolor)
        display_selected(fig, selected, color)
        plt.show()
    except Exception as err:
        print(f"{err.__class__.__name__}: {err}", file=sys.stderr)


if __name__ == "__main__":
    main()
