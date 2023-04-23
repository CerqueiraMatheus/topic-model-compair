# Import libraries

import nltk
from octis.models.CTM import CTM
from octis.models.ETM import ETM
from octis.models.HDP import HDP
from octis.models.LDA import LDA
from octis.models.LSI import LSI
from octis.models.NMF import NMF
from octis.models.ProdLDA import ProdLDA
from octis.dataset.dataset import Dataset
from octis.models.NeuralLDA import NeuralLDA
from octis.optimization.optimizer import Optimizer
from skopt.space.space import Real, Categorical, Integer
from octis.evaluation_metrics.coherence_metrics import Coherence
from octis.evaluation_metrics.diversity_metrics import TopicDiversity

from custom_metric import TDCI

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="darkgrid")
sns.set_context("paper")


def symmetrize_y_axis(axes, y_min, y_max):
    # y_max = np.abs(axes.get_ylim()).max()
    axes.set_ylim(ymin=y_min, ymax=y_max)


def optimize(model: any, search_space: dict, save_path: str, csv_path: str):
    """
    Optimize a given model in a search space.

    """

    # Create optimizer
    optimizer = Optimizer()

    # Optimize
    ctm_optimization_result = optimizer.optimize(
        model,
        dataset,
        tdci,
        search_space,
        random_state=42,
        model_runs=model_runs,
        save_models=True,
        extra_metrics=[topic_coherence, topic_diversity],  # to keep track of other metrics
        number_of_call=optimization_runs,
        save_path=save_path,
    )

    # Export csv
    ctm_optimization_result.save_to_csv(csv_path)


def optimize_ctm():
    # Define search space
    ctm_search_space = {
        "n_topics": list(range(10, 41, 10)),
        "num_layers": Categorical({1, 2, 3, 4}),
        "num_neurons": Categorical({10, 30, 50, 100, 200, 300}),
        "activation": Categorical(
            {"softplus", "relu", "sigmoid", "tanh", "leakyrelu", "rrelu", "elu", "selu"}
        ),
        "solver": Categorical({"adam", "sgd"}),
        "dropout": Real(0.0, 0.95),
    }

    # Define model and optimize
    model_ctm = CTM()
    optimize(
        model_ctm,
        ctm_search_space,
        "tunning/test_ctm//",
        "tunning_results/results_ctm.csv",
    )

def optimize_etm():
    etm_search_space = {
        "n_topics": list(range(10, 41, 10)),
        "optimizer": Categorical({"adam", "adagrad", "adadelta", "rmsprop", "asgd", "sgd"}),
        "t_hidden_size": Integer(400, 1000),
        "rho": Integer(200, 600),
        "num_neurons": Categorical({100, 200, 300}),
        "activation": Categorical({'sigmoid', 'relu', 'softplus'}), 
        "dropout": Real(0.0, 0.95)
    }

    model_etm = ETM(device="gpu")
    optimize(model_etm, 
         etm_search_space, 
         "tunning/test_etm//", 
         "tunning_results/results_etm.csv"
         )

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
        x=execs, y=model_runs, color=sns.color_palette("tab10")[0], label="TC mean"
    )
    ax1.set_ylabel("Topic Coherence")
    symmetrize_y_axis(ax1, -0.6, 0.9)
    ax1.get_legend().remove()

    # Define topic diversity plot
    plt.xlabel("Execution")
    plt.title("Performance along search space - " + alg_name)
    ax2 = ax1.twinx()
    ax2 = sns.lineplot(
        x=execs, y=topic_div, color=sns.color_palette("tab10")[1], label="TD mean"
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
        df.sort_values(by="Mean(model_runs)", ascending=False).head(1).to_numpy()
    )

    # Print params
    print("Best params")
    for i in list(range(len(df.columns))):
        print(df.columns[i] + " = " + str(best_conf[0][i]))


if __name__ == "__main__":
    import os

    os.chdir(os.getenv("HOME"))
    os.chdir("octis-compair")

    model_runs = 1
    optimization_runs = 5

    dataset = Dataset()
    # dataset.load_custom_dataset_from_folder("pl_3723_2019")
    dataset.fetch_dataset("20NewsGroup")

    # Define metric for tunning
    topic_coherence = Coherence(texts=dataset.get_corpus())

    # Define other metrics
    topic_diversity = TopicDiversity(topk=10)

    tdci = TDCI(texts=dataset.get_corpus())

    # optimize_ctm()
    # optimize_etm()

    visualize_model_along_exec("tunning_results/results_etm.csv", "ETM")
