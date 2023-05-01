import os
import sys
import nltk
import pandas as pd

from tqdm import tqdm
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

from typing import List

delimiter = ';'
is_acordaos = False
col_content = 'des_conteudo'
col_name = 'nom_titulo_formulario_publicado'


def set_acordaos():
    """
    """
    delimiter = ','
    is_acordaos = True
    col_content = 'formatado'
    col_name = 'codigos_movimentos_temas'


def get_sentences_from_pl(csv_path: str, col: str):
    """
    """

    dataset = pd.read_csv(csv_path, delimiter=';')
    dataset_pl = dataset.loc[dataset[col_name] == col]

    return list(dataset_pl[col_content])


def get_stopwords():
    """
    """

    stop = set(stopwords.words('portuguese'))
    stop.remove("não")

    if is_acordaos:
        stop.update(
            ["registro", "acórdão", "vistos",
             "relatados", "discutidos", "autos",
                "apelação", "cível", "nº", "comarca"])

    return stop


def create_vocabulary(sentences: List[str], path: str = None):
    """
    """

    words = set()

    for sentence in tqdm(sentences):
        if isinstance(sentence, str):
            cur_words = [word.lower()
                         for word in
                         nltk.word_tokenize(sentence) if
                         (word.isalnum() and word.lower() not in get_stopwords())]
        words.update(cur_words)

    with open(path + "vocabulary.txt", 'w') as vocabulary:
        for word in words:
            vocabulary.write(word)
            vocabulary.write('\n')

    return


def create_corpus(sentences: List[str], path: str = None):
    """
    """
    ren_sentences = []

    # Create valid sentences
    for sentence in tqdm(sentences):
        if (isinstance(sentence, str)):
            cur_words = [word.lower()
                         for word in
                         nltk.word_tokenize(sentence) if
                         (word.isalpha() and word.lower() not in get_stopwords())]

            new_sentence = ' '.join(cur_words)
            if (new_sentence.strip() != ""):
                ren_sentences.append(new_sentence)

    # Split train, test and validation
    X_train, X_rem = train_test_split(
        ren_sentences, train_size=0.7, random_state=42)
    X_test, X_val = train_test_split(X_rem, test_size=0.5, random_state=42)

    # Add labels
    train_list = [[sentence, 'train'] for sentence in X_train]
    test_list = [[sentence, 'test'] for sentence in X_test]
    val_list = [[sentence, 'val'] for sentence in X_val]

    # Concat lists
    concat_list = train_list + test_list + val_list

    corpus_df = pd.DataFrame(concat_list, columns=[
                             'text', 'train']).replace('"', '', regex=True)
    corpus_df.to_csv(path + "corpus.tsv", sep='\t', index=False, header=False)

    return


if __name__ == "__main__":
    if (len(sys.argv) != 4):
        print("Usage: python3 preprocess.py input_csv_path.csv output_folder_path 0/")
        exit(1)

    if not os.path.exists(sys.argv[2]):
        os.makedirs(sys.argv[2])

    if (int(sys.argv[3]) == 1):
        set_acordaos()

    sentences = get_sentences_from_pl(sys.argv[1], "PEC 471/2005")
    create_vocabulary(sentences, sys.argv[2])
    create_corpus(sentences, sys.argv[2])

    exit(0)
