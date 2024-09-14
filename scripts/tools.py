from datetime import datetime as date
from typing import Generator

from pandas import Index
from pandas import DataFrame
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.widgets import Button
from matplotlib.backend_bases import KeyEvent


def get_age(string: str) -> float:
    return (date.today() - date.strptime(string, "%Y-%m-%d")).days / 365


def close_callback(event: KeyEvent) -> None:
    """Callback function to close a plot by pressing escape."""
    if event.key == "escape":
        plt.close()


def shorten(string: str, size: int = 15) -> str:
    """Shortens long names for buttons."""
    if len(string) > size:
        if ' ' in string:
            new_string = ""
            for c in string:
                if c.isupper():
                    new_string += c
            return f"{new_string:>.{size}s}"
    return f"{string:>.{size}s}"


def button_gen(fig: Figure, idx: Index) -> Generator:
    """Generator for a list of buttons."""
    width = 1 / len(idx)
    for i in range(len(idx)):
        b_axis = fig.add_axes([i * width + 0.0015, 0.03, width - 0.003, 0.05])
        yield Button(b_axis, shorten(idx[i]), color=(1, 1, 1, 1),
                     hovercolor=(0, 0, 0, 0.5))


def house_gen(data: DataFrame, course: str, house: str) -> Generator:
    """Generator for a list of houses datas."""
    i = 0
    while i < data.shape[0]:
        value = data.at[i, course]
        if data.at[i, "Hogwarts House"] == house:
            yield value
        i += 1


def courses_gen(raw_data: DataFrame, data: DataFrame) -> Generator:
    """Generator for a list of dictionaries holding houses datas."""
    for course in data.columns:
        yield (course,
               {"Hufflepuff": list(house_gen(raw_data, course, "Hufflepuff")),
                "Ravenclaw": list(house_gen(raw_data, course, "Ravenclaw")),
                "Gryffindor": list(house_gen(raw_data, course, "Gryffindor")),
                "Slytherin": list(house_gen(raw_data, course, "Slytherin"))})
