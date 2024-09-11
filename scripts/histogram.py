import argparse
import pandas
import sys
import matplotlib.pyplot as plt


def scores_houses(data: pandas.DataFrame, course: str):
    """Returns two lists: one with scores and one with respespective houses."""
    score = []
    house = []
    for i in data.Index:
        if data.at[i, course] == data.at[i, course]:
            match data.at[i, "Hogwarts House"]:
                case "Ravenclaw":
                    score.append(data.at[i, course])
                    house.append(0)
                case "Slytherin":
                    score.append(data.at[i, course])
                    house.append(1)
                case "Gryffindor":
                    score.append(data.at[i, course])
                    house.append(2)
                case "Hufflepuff":
                    score.append(data.at[i, course])
                    house.append(3)
                case _:
                    pass
    return (score, house)


def main() -> None:
    """Display an histogram of the given dataframe."""
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("data", help="Dataset to be read.")
        data = pandas.read_csv(parser.parse_args().data)
        columns = list(data.select_dtypes(float).columns)
        fig, axis = plt.subplots(len(columns))
        fig.set_size_inches(15, 9)
        fig.set_facecolor((0, 0, 0))
        for i in range(len(columns)):
            axis[i].hist2d(*scores_houses(data, columns[i]), bins=(100, 4))
            axis[i].set_axis_off()
            axis[i].set_title(columns[i], x=-0.02, y=0.2, loc="right",
                              color=(1, 1, 1))
        plt.subplots_adjust(left=0.2, right=0.9, top=0.99, bottom=0.01,
                            hspace=1)
        plt.show()
    except Exception as err:
        print(f"{type(err).__name__}: {err}", file=sys.stderr)


if __name__ == "__main__":
    main()
