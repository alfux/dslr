import argparse
import sys
import pandas
import matplotlib.pyplot as plt
import matplotlib.backend_bases
from datetime import datetime


def get_age(date: str):
    return (datetime.today() - datetime.strptime(date, "%Y-%m-%d")).days / 365


def main() -> None:
    """Displays a scatter plot to find the two similar features."""
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("data", help="Dataset to be read.")
        raw_data = pandas.read_csv(parser.parse_args().data)
        data = raw_data.select_dtypes(float)
        ages = [get_age(x) for x in raw_data.Birthday]
        selected_data = {course: None for course in data.columns}
        (fig, axis) = plt.subplots(num="Scatter plot")
        fig.set_size_inches(15, 9)

        def plot_selected_data() -> None:
            """Adds every selected data to the plot."""
            for (course, color) in selected_data.items():
                axis.plot(ages,
                          data[course],
                          'o',
                          label=course,
                          color=color,
                          markersize=5)
            axis.set_xlabel("Ages")
            axis.set_ylabel("Scores")

        plot_selected_data()
        leg = fig.legend(loc="outside upper center", ncols=4, borderaxespad=1)
        for (text, legend) in zip(leg.get_texts(), leg.get_lines()):
            text.set_picker(5)
            text.set_label(text.get_text())
            selected_data[text.get_text()] = legend.get_color()
        saved_data = dict(selected_data)

        def on_click(event: matplotlib.backend_bases.PickEvent) -> None:
            """Event callback to remove or add plot on legend click"""
            course = event.artist.get_label()
            if course not in data.columns:
                return
            if course in selected_data:
                del selected_data[course]
                event.artist.set_alpha(0.2)
            else:
                selected_data[course] = saved_data[course]
                event.artist.set_alpha(1)
            axis.clear()
            plot_selected_data()
            fig.canvas.draw()

        fig.canvas.mpl_connect('pick_event', on_click)
        plt.show()
    except Exception as err:
        print(f"{type(err).__name__}: {err}", file=sys.stderr)


if __name__ == "__main__":
    main()
