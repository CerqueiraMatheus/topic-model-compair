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

from custom.metrics.TDCI import TDCI
# from custom_models import CustomTop2Vec

base_csv_path = "tunning/csv/"
base_res_path = "tunning/res/"

def __optimize(model: any, search_space: dict, save_path: str, csv_path: str):
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
        # to keep track of other metrics
        extra_metrics=[topic_coherence, topic_diversity],
        number_of_call=optimization_runs,
        save_path=save_path,
    )

    # Export csv
    ctm_optimization_result.save_to_csv(csv_path)


def optimize_ctm(n_topics: int = 10):
    # Define search space
    ctm_search_space = {
        "n_topics": [n_topics],
        "num_layers": Categorical({1, 2, 3, 4}),
        "num_neurons": Categorical({10, 30, 50, 100, 200, 300}),
        "activation": Categorical(
            {"softplus", "relu", "sigmoid", "tanh",
                "leakyrelu", "rrelu", "elu", "selu"}
        ),
        "solver": Categorical({"adam", "sgd"}),
        "dropout": Real(0.0, 0.95),
        "inference_type": Categorical({"zeroshot", "combined"})
    }

    # Define model and optimize
    model_ctm = CTM()
    __optimize(
        model_ctm,
        ctm_search_space,
        base_res_path + "test_ctm" + str(n_topics) + "//",
        base_csv_path + "results_ctm" + str(n_topics) + ".csv"
    )


def optimize_etm(n_topics: int = 10):
    etm_search_space = {
        "num_topics": [n_topics],
        "optimizer": Categorical({"adam", "adagrad", "adadelta", "rmsprop", "asgd", "sgd"}),
        "t_hidden_size": Integer(400, 1000),
        "rho": Integer(200, 600),
        "num_neurons": Categorical({100, 200, 300}),
        "activation": Categorical({'sigmoid', 'relu', 'softplus'}),
        "dropout": Real(0.0, 0.95)
    }

    model_etm = ETM(device="gpu")
    __optimize(model_etm,
               etm_search_space,
               base_res_path + "test_etm" + str(n_topics) + "//",
               base_csv_path + "results_etm" + str(n_topics) + ".csv"
               )


def optimize_lda(n_topics: int = 10):
    lda_search_space = {
        "num_topics": [n_topics],

    }

    model_lda = LDA()
    __optimize(model_lda,
               lda_search_space,
               base_res_path + "test_lda" + str(n_topics) + "//",
               base_csv_path + "results_lda" + str(n_topics) + ".csv"
               )


def optimize_lsi(n_topics: int = 10):
    lsi_search_space = {
        "num_topics": [n_topics],

    }

    model_lsi = LSI()
    __optimize(model_lsi,
               lsi_search_space,
               base_res_path + "test_lsi" + str(n_topics) + "//",
               base_csv_path + "results_lsi" + str(n_topics) + ".csv"
               )


def optimize_nmf(n_topics: int = 10):
    nmf_search_space = {
        "num_topics": [n_topics],
    }

    model_nmf = NMF()
    __optimize(model_nmf,
               nmf_search_space,
               base_res_path + "test_nmf" + str(n_topics) + "//",
               base_csv_path + "results_nmf" + str(n_topics) + ".csv"
               )


def optimize_prodlda(n_topics: int = 10):
    prodlda_search_space = {
        "num_topics": [n_topics],
    }

    model_prodlda = ProdLDA()
    __optimize(model_prodlda,
               prodlda_search_space,
               base_res_path + "test_prodlda" + str(n_topics) + "//",
               base_csv_path + "results_prodlda" + str(n_topics) + ".csv"
               )


def optimize_neurallda(n_topics: int = 10):
    neurallda_search_space = {
        "num_topics": [n_topics],
    }

    model_neurallda = NeuralLDA()
    __optimize(model_neurallda,
               neurallda_search_space,
               base_res_path + "test_neurallda" + str(n_topics) + "//",
               base_csv_path + "results_neurallda" + str(n_topics) + ".csv"
               )


def optimize_hdp():
    hdp_search_space = {
    }

    model_hdp = HDP()
    __optimize(model_hdp,
               hdp_search_space,
               base_res_path + "test_hdp//",
               base_csv_path + "results_hdp.csv"
               )


# TODO: implementation of the model is not complete
def optimize_custom_top2vec():
    custom_top2vec_search_space = {
    }

    model_custom_top2vec = CustomTop2Vec()
    __optimize(model_custom_top2vec,
               custom_top2vec_search_space,
               base_res_path + "test_custom_top2vec//",
               base_csv_path + "results_custom_top2vec.csv"
               )


# TODO: implementation of the model is not complete
def optimize_bertopic():
    bertopic_search_space = {
    }

    model_bertopic = None
    __optimize(model_bertopic,
               bertopic_search_space,
               base_res_path + "test_bertopic//",
               base_csv_path + "results_bertopic.csv"
               )


if __name__ == "__main__":
    import os

    # os.chdir(os.getenv("HOME"))
    # os.chdir("octis-compair")

    model_runs = 1
    optimization_runs = 2

    dataset = Dataset()
    # dataset.load_custom_dataset_from_folder("pl_3723_2019")
    dataset.fetch_dataset("20NewsGroup")

    # Define metric for tunning
    topic_coherence = Coherence(texts=dataset.get_corpus())

    # Define other metrics
    topic_diversity = TopicDiversity(topk=10)

    tdci = TDCI(texts=dataset.get_corpus())

    # Run optimizer for models depending on n_topics
    for n_topics in [10]:
        # optimize_ctm(n_topics)
        optimize_etm(n_topics)
        optimize_lda(n_topics)
        optimize_lsi(n_topics)
        optimize_nmf(n_topics)
        optimize_prodlda(n_topics)
        optimize_neurallda(n_topics)

    # Models that doesn't require n_topics
    optimize_hdp()
    # optimize_custom_top2vec()
    # optimize_bertopic()
