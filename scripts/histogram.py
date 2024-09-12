import argparse
import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import typing as tp
import matplotlib.figure as mtp
import matplotlib.backend_bases as mbtp


def shorten(string: str) -> str:
    """Shortens long names for buttons."""
    if len(string) > 15:
        new_string = ""
        for c in string:
            if c.isupper():
                new_string += c
        return new_string
    return string


def button_gen(fig: mtp.Figure, idx: pd.Index) -> tp.Generator:
    """Generator for a list of buttons."""
    width = 1 / len(idx)
    for i in range(len(idx)):
        b_axis = fig.add_axes([i * width + 0.0015, 0.03, width - 0.003, 0.05])
        yield (Button(b_axis, shorten(idx[i]), color=(1, 1, 1),
                      hovercolor=(0, 0, 0, 0.5)))


def house_gen(data: pd.DataFrame, course: str, house: str) -> tp.Generator:
    """Generator for a list of houses datas."""
    i = 0
    while i < data.shape[0]:
        value = data.at[i, course]
        if value == value and data.at[i, "Hogwarts House"] == house:
            yield (value)
        i += 1


def course_gen(raw_data: pd.DataFrame, data: pd.DataFrame) -> tp.Generator:
    """Generator for a list of dictionaries holding houses datas."""
    for course in data.columns:
        yield ({"Hufflepuff": list(house_gen(raw_data, course, "Hufflepuff")),
                "Ravenclaw": list(house_gen(raw_data, course, "Ravenclaw")),
                "Gryffindor": list(house_gen(raw_data, course, "Gryffindor")),
                "Slytherin": list(house_gen(raw_data, course, "Slytherin"))})


def display_course(fig: mtp.Figure, data: dict, key: str, color: dict) -> None:
    """Refresh the histogram."""
    ax = fig.get_axes()[-1]
    ax.clear()
    ax.set_title(key)
    for (sub_key, values) in data.items():
        ax.hist(values, color=color[sub_key], label=sub_key, bins=50)
    fig.legend(fancybox=True, shadow=True)
    fig.canvas.draw()


def main() -> None:
    """Display an histogram of the given dataframe."""
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("data", help="Dataset to be read.")
        raw_data = pd.read_csv(parser.parse_args().data)
        data = raw_data.select_dtypes(float)
        course = list(course_gen(raw_data, data))
        color = {"Ravenclaw": (0.3, 0, 1, 1), "Slytherin": (0, 0.5, 0, 1),
                 "Gryffindor": (1, 0, 0, 1), "Hufflepuff": (1, 0.7, 0, 1)}
        fig = plt.figure("Histograms", facecolor=(0.8, 0.8, 0.8))
        fig.set_size_inches(15, 9)

        def create_callback(i: int) -> tp.Callable[[mbtp.Event], None]:
            """Creates a callback function relative to i's value."""
            return (lambda event: display_course(
                fig, course[i], data.columns[i], color))

        buttons: list[Button] = list(button_gen(fig, data.columns))
        for i in range(len(buttons)):
            buttons[i].on_clicked(create_callback(i))
        fig.add_subplot(position=(0.125, 0.15, 0.75, 0.75))
        display_course(fig, course[10], data.columns[10], color)
        plt.show()
    except Exception as err:
        print(f"{type(err).__name__}: {err}", file=sys.stderr)


if __name__ == "__main__":
    main()
