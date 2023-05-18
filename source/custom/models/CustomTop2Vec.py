from top2vec import Top2Vec
import gensim.corpora as corpora
from octis.models.model import AbstractModel


class CustomTop2Vec(AbstractModel):

    id2word = None
    id_corpus = None
    hyperparameters = {}
    use_partitions = True
    update_with_test = False

    def __init__(self, min_count=50, topic_merge_delta=0.1, ngram_vocab=False, ngram_vocab_args=None, embedding_model='doc2vec', embedding_model_path=None, speed='learn', use_corpus_file=False, document_ids=None, keep_documents=True, workers=None, tokenizer=None, use_embedding_model_tokenizer=False, umap_args=None, hdbscan_args=None, verbose=True, num_topics=None):

        super().__init__()
        self.hyperparameters['min_count'] = min_count
        self.hyperparameters['topic_merge_delta'] = topic_merge_delta
        self.hyperparameters['ngram_vocab'] = ngram_vocab
        self.hyperparameters['ngram_vocab_args'] = ngram_vocab_args
        self.hyperparameters['embedding_model'] = embedding_model
        self.hyperparameters['embedding_model_path'] = embedding_model_path
        self.hyperparameters['speed'] = speed
        self.hyperparameters['use_corpus_file'] = use_corpus_file
        self.hyperparameters['document_ids'] = document_ids
        self.hyperparameters['keep_documents'] = keep_documents
        self.hyperparameters['workers'] = workers
        self.hyperparameters['tokenizer'] = tokenizer
        self.hyperparameters['use_embedding_model_tokenizer'] = use_embedding_model_tokenizer
        self.hyperparameters['umap_args'] = umap_args
        self.hyperparameters['hdbscan_args'] = hdbscan_args
        self.hyperparameters['verbose'] = verbose

        self._model = None

    def info(self):
        """
        Returns model informations
        """
        return {
            "name": "Top2Vec"
        }

    def train_model(self, dataset, hyperparams={}, top_words=10):
        """
        Train the model and return output

        Parameters
        ----------
        dataset : dataset to use to build the model
        hyperparameters : hyperparameters to build the model
        top_words : if greather than 0 returns the most significant words
                 for each topic in the output
                 Default True

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

        self._prepare_model(train_corpus)

        result["topics"] = self._get_topics()

        print(self.id2word)

        topics_old = [list(topic[:top_words]) for topic in result["topics"]]

        all_words = [word for words in train_corpus for word in words]
        topics = []
        for topic in topics_old:
            words = []
            for word in topic:
                if word in all_words:
                    words.append(word)
                else:
                    print(f"error: {word}")
                    words.append(all_words[0])
            topics.append(words)
        result["topics"] = topics

        return result

    def _prepare_model(self, dataset):
        self._model = Top2Vec(
            documents=[' '.join(doc) for doc in dataset],
            embedding_model="doc2vec"
        )

    def _get_topics(self):
        return self._model.get_topics()[0]

    def _get_topic_word_matrix(self):
        return self._model.get_topics()[1]

    def _get_topic_document_matrix(self):
        return self._model.get_documents_topics(doc_ids=self._model.document_ids,)[1]

    def _get_document_topic_tuples(self):
        return [tuple(l) for l in self._model.get_documents_topics(doc_ids=self._model.document_ids, num_topics=self._model.get_num_topics())[0]]
