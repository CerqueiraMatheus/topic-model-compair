import os
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def symmetrize_y_axis(axes, y_min, y_max):
    axes.set_ylim(ymin=y_min, ymax=y_max)


def visualize_model_along_exec(alg_name: str, dataset_name: str):

    dfs = []
    n_topics = [10, 20, 30, 40, 50]

    for n_topic in n_topics:
        dfs.append(list(pd.read_csv("tunning/csv/" + alg_name + "-" +
                   dataset_name + "-" + str(n_topic) + ".csv")["Mean(model_runs)"]))
        print(pd.read_csv("tunning/csv/" + alg_name + "-" + dataset_name + "-" + str(n_topic) + ".csv").sort_values(
            by="Mean(model_runs)", ascending=False).head(1))

    str_topics = [str(x) for x in n_topics]
    plt.figure(figsize=(4,2.5))
    sns.lineplot(data=dfs)
    plt.xlabel("Optimization Run")
    plt.ylabel("Mean TDCI")
    plt.title("TDCI - " + alg_name.upper() +
              " - " + dataset_name.replace('_', ' ').upper())
    plt.legend(str_topics, title="Topics")
    plt.savefig("plots/" + dataset_name + "-" +
                alg_name + ".svg", format="svg", dpi=300, bbox_inches="tight")
    plt.clf()


if __name__ == "__main__":
    # os.chdir(os.getenv("HOME"))
    # os.chdir("octis-compair")

    # fm = matplotlib.font_manager
    # fm._get_fontconfig_fonts.cache_clear()

    sns.set(style="darkgrid")
    sns.set_context("paper")
    plt.rcParams["font.size"] = 8
    plt.rcParams["font.family"] = "Barlow Medium"

    for tm in ["ctm", "etm", "lda", "lsi", "nmf", "neurallda", "prodlda"]:
        print(tm)
        visualize_model_along_exec(tm, "pl_3723_2019")
        print("\n")
    # exit(0)
