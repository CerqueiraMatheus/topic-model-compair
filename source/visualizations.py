import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="darkgrid")
sns.set_context("paper")


def symmetrize_y_axis(axes, y_min, y_max):
    axes.set_ylim(ymin=y_min, ymax=y_max)


def visualize_model_along_exec(alg_name: str, dataset_name: str):
    
    dfs = []
    n_topics = [10, 20, 30, 40, 50]

    for n_topic in n_topics:
        dfs.append(list(pd.read_csv("tunning/csv/" + alg_name + "-" + dataset_name + "-" + str(n_topic) + ".csv")["Mean(model_runs)"]))

    str_topics = [str(x) for x in n_topics]
    sns.lineplot(dfs)
    plt.xlabel("Optimization Run")
    plt.ylabel("Mean TDCI")
    plt.legend(str_topics)
    plt.savefig("plots/" + dataset_name + "-" + alg_name + ".svg", format="svg", dpi=300)
    plt.clf()

if __name__ == "__main__":
    os.chdir(os.getenv("HOME"))
    os.chdir("octis-compair")

    for tm in ["ctm", "etm", "lda", "lsi", "nmf", "neurallda", "prodlda"]:
        print(tm)
        visualize_model_along_exec(tm, "pec_471_2005")
        print("\n")
    exit(0)