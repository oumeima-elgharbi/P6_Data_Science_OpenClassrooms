import warnings
warnings.filterwarnings(action="ignore")

from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.layers import *
from keras.models import Model
import gensim

import numpy as np


def create_w2v_model(sentences, w2v_min_count, w2v_size, w2v_window, w2v_epochs):
    """
    # Création et entraînement du modèle Word2Vec

    :param sentences:
    :param w2v_min_count:
    :param w2v_size:
    :param w2v_window:
    :param w2v_epochs:
    :return:
    """
    print("Build & train Word2Vec model ...")
    w2v_model = gensim.models.Word2Vec(min_count=w2v_min_count, window=w2v_window,
                                       vector_size=w2v_size,
                                       seed=42,
                                       workers=1)
    #                                                workers=multiprocessing.cpu_count())
    w2v_model.build_vocab(sentences)
    w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=w2v_epochs)
    model_vectors = w2v_model.wv
    w2v_words = model_vectors.index_to_key
    print("Vocabulary size: %i" % len(w2v_words))
    print("Word2Vec trained")
    return w2v_words, model_vectors


def tokenize_sentences(sentences, maxlen):
    """
    # Préparation des sentences (tokenization)

    :param sentences:
    :param maxlen:
    :return:
    """

    print("Fit Tokenizer ...")
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sentences)
    x_sentences = pad_sequences(tokenizer.texts_to_sequences(sentences),
                                maxlen=maxlen,
                                padding='post')

    num_words = len(tokenizer.word_index) + 1
    print("Number of unique words: %i" % num_words)
    return x_sentences, tokenizer


def create_embedding_matrix(w2v_words, model_vectors, tokenizer):
    """
    # Création de la matrice d'embedding

    :param w2v_words:
    :param model_vectors:
    :param tokenizer:
    :return:
    """

    print("Create Embedding matrix ...")
    w2v_size = 300
    word_index = tokenizer.word_index
    vocab_size = len(word_index) + 1
    embedding_matrix = np.zeros((vocab_size, w2v_size))
    i = 0
    j = 0

    for word, idx in word_index.items():
        i += 1
        if word in w2v_words:
            j += 1
            embedding_vector = model_vectors[word]
            if embedding_vector is not None:
                embedding_matrix[idx] = model_vectors[word]

    word_rate = np.round(j / i, 4)
    print("Word embedding rate : ", word_rate)
    print("Embedding matrix: %s" % str(embedding_matrix.shape))
    return embedding_matrix, vocab_size


def create_embedding_model(x_sentences, maxlen, vocab_size, w2v_size, embedding_matrix):
    """
    # Création du modèle

    :param x_sentences:
    :param maxlen:
    :param vocab_size:
    :param w2v_size:
    :param embedding_matrix:
    :return:
    """

    input = Input(shape=(len(x_sentences), maxlen), dtype='float64')
    word_input = Input(shape=(maxlen,), dtype='float64')
    word_embedding = Embedding(input_dim=vocab_size,
                               output_dim=w2v_size,
                               weights=[embedding_matrix],
                               input_length=maxlen)(word_input)
    word_vec = GlobalAveragePooling1D()(word_embedding)
    embed_model = Model([word_input], word_vec)

    embed_model.summary()

    return embed_model
