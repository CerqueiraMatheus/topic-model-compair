from bertopic import BERTopic
from typing import List, Tuple, Union, Mapping, Any
import hdbscan
from umap import UMAP
import gensim.corpora as corpora
from octis.models.model import AbstractModel
from sklearn.feature_extraction.text import CountVectorizer


class CustomBERTopic(AbstractModel):
    
    id2word = None
    id_corpus = None
    hyperparameters = {}
    use_partitions = True
    update_with_test = False

    def __init__(self,
                 language: str = "english",
                 top_n_words: int = 10,
                 n_gram_range: Tuple[int, int] = (1, 1),
                 min_topic_size: int = 10,
                 nr_topics: Union[int, str] = None,
                 low_memory: bool = False,
                 calculate_probabilities: bool = False,
                 diversity: float = None,
                 seed_topic_list: List[List[str]] = None,
                 embedding_model=None,
                 umap_model: UMAP = None,
                 hdbscan_model: hdbscan.HDBSCAN = None,
                 vectorizer_model: CountVectorizer = None,
                 verbose: bool = False):
        super().__init__()
        self.hyperparameters['language'] = language
        self.hyperparameters['top_n_words'] = top_n_words
        self.hyperparameters['n_gram_range'] = n_gram_range
        self.hyperparameters['min_topic_size'] = min_topic_size
        self.hyperparameters['nr_topics'] = nr_topics
        self.hyperparameters['low_memory'] = low_memory
        self.hyperparameters['calculate_probabilities'] = calculate_probabilities
        self.hyperparameters['diversity'] = diversity
        self.hyperparameters['seed_topic_list'] = seed_topic_list
        self.hyperparameters['embedding_model'] = embedding_model
        self.hyperparameters['umap_model'] = umap_model
        self.hyperparameters['hdbscan_model'] = hdbscan_model
        self.hyperparameters['vectorizer_model'] = vectorizer_model
        self.hyperparameters['verbose'] = verbose

        self._model = None

    def info(self):
        """
        Returns model informations
        """
        return {
            "name": "BERTopic"
        }
    
    def train_model(self, dataset, hyperparams={}, top_words=10):
        """
        Train the model and return output

        Parameters
        ----------
        dataset : dataset to use to build the model
        hyperparameters : hyperparameters to build the model

        Returns
        -------
        result : dictionary with up to 3 entries,
                 'topics', 'topic-word-matrix' and
                 'topic-document-matrix'
        """

        if hyperparams is None:
            hyperparams = {}

        if self.use_partitions:
            train_corpus, test_corpus = dataset.get_partitioned_corpus(
                use_validation=False)
        else:
            train_corpus = dataset.get_corpus()

        if self.id2word is None:
            self.id2word = corpora.Dictionary(dataset.get_corpus())

        if self.id_corpus is None:
            self.id_corpus = [self.id2word.doc2bow(document)
                              for document in train_corpus]

        hyperparams["corpus"] = self.id_corpus
        hyperparams["id2word"] = self.id2word
        self.hyperparameters.update(hyperparams)

        result = {}

        self._prepare_model()
        topics, _ = self._model.fit_transform(documents=[' '.join(doc) for doc in train_corpus])
        
        all_words = [word for words in train_corpus for word in words]
        bertopic_topics = [
            [
                vals[0] if vals[0] in all_words else all_words[0]
                for vals in self._model.get_topic(i)[:top_words]
            ]
            for i in range(len(set(topics)) - 1)
        ]

        result['topics'] = bertopic_topics

        return result

    def _prepare_model(self):
        self._model = BERTopic(
            language="multilingual",
            embedding_model='all-mpnet-base-v2'
        )
    

        