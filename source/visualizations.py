import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="darkgrid")
sns.set_context("paper")


def symmetrize_y_axis(axes, y_min, y_max):
    # y_max = np.abs(axes.get_ylim()).max()
    axes.set_ylim(ymin=y_min, ymax=y_max)


def visualize_model_along_exec(path: str, alg_name: str):
    # Read dataset
    df = pd.read_csv(path)

    # Get model_runs
    model_runs = df["Mean(model_runs)"]
    topic_div = df["Coherence(not optimized)"]

    # Define executions
    execs = list(range(len(model_runs)))

    fig, ax = plt.subplots()
    # fig = plt.figure()

    # Define axis and title
    plt.xlabel("Execução")
    plt.title("Desempenho ao longo do espaço de busca - " + alg_name)

    # Define topic coherence plot
    ax1 = sns.lineplot(
        x=execs, y=model_runs, color=sns.color_palette("tab10")[
            0], label="TC mean"
    )
    ax1.set_ylabel("Topic Coherence")
    symmetrize_y_axis(ax1, -0.6, 0.9)
    ax1.get_legend().remove()

    # Define topic diversity plot
    plt.xlabel("Execution")
    plt.title("Performance along search space - " + alg_name)
    ax2 = ax1.twinx()
    ax2 = sns.lineplot(
        x=execs, y=topic_div, color=sns.color_palette("tab10")[
            1], label="TD mean"
    )
    ax2.grid(False)
    ax2.set_ylabel("Topic Diversity")
    symmetrize_y_axis(ax2, 0.2, 1.0)
    ax2.get_legend().remove()

    # Legend
    fig.legend(
        loc="upper right", bbox_to_anchor=(0.95, 0.2), bbox_transform=ax.transAxes
    )

    # Export
    plt.savefig("plots/" + alg_name + ".svg", format="svg", dpi=300)

    # Show
    plt.show()

    # Get best
    best_conf = (
        df.sort_values(by="Mean(model_runs)",
                       ascending=False).head(1).to_numpy()
    )

    # Print params
    print("Best params")
    for i in list(range(len(df.columns))):
        print(df.columns[i] + " = " + str(best_conf[0][i]))


if __name__ == "__main__":
    visualize_model_along_exec("tunning_results/results_etm.csv", "ETM")
    exit(0)