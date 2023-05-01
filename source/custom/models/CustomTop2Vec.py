import numpy as np
from top2vec import Top2Vec
import gensim.corpora as corpora
from octis.models.model import AbstractModel
import octis.configuration.defaults as defaults


class CustomTop2Vec(AbstractModel):
    def __init__(self, min_count=50, topic_merge_delta=0.1, ngram_vocab=False, ngram_vocab_args=None, embedding_model='doc2vec', embedding_model_path=None, speed='learn', use_corpus_file=False, document_ids=None, keep_documents=True, workers=None, tokenizer=None, use_embedding_model_tokenizer=False, umap_args=None, hdbscan_args=None, verbose=True, num_topics=10):

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
        self.hyperparameters['num_topics'] = num_topics

    def info(self):
        """
        Returns model informations
        """
        return {
            "name": "Top2Vec"
        }

    def hyperparameters_info(self):
        """
        Returns hyperparameters informations
        """
        return defaults.LSI_hyperparameters_info

    def train_model(self, dataset):
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

        result = {}

        docs, _ = dataset.get_partitioned_corpus(use_validation=False)

        new_docs = []
        for doc in docs:
            new_docs.append(' '.join(doc))

        self.hyperparameters["documents"] = new_docs
        print(self.hyperparameters["documents"])
        top2vec = Top2Vec(
            documents=self.hyperparameters["documents"],
            min_count=self.hyperparameters["min_count"],
            embedding_model=self.hyperparameters["embedding_model"],
            embedding_model_path=self.hyperparameters["embedding_model_path"],
            speed=self.hyperparameters["speed"],
            use_corpus_file=self.hyperparameters["use_corpus_file"],
            document_ids=self.hyperparameters["document_ids"],
            keep_documents=self.hyperparameters["keep_documents"],
            workers=self.hyperparameters["workers"],
            tokenizer=self.hyperparameters["tokenizer"],
            use_embedding_model_tokenizer=self.hyperparameters["use_embedding_model_tokenizer"],
            umap_args=self.hyperparameters["umap_args"],
            hdbscan_args=self.hyperparameters["hdbscan_args"],
            verbose=self.hyperparameters["verbose"]
        )

        execution = top2vec.get_topics(
            num_topics=self.hyperparameters["num_topics"])
        
        result["topics"] = execution[0]
        result["topic-word-matrix"] = execution[1]

        return result


if __name__ == "__main__":
    model = CustomTop2Vec()
    print(model.info())
