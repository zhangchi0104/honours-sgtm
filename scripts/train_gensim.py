import gensim
from typing import Iterable
import numpy as np
import pandas as pd
from utils.visualize import visualize_results
from utils.io import load_seed, load_dataset
import argparse

import logging


def learn_embeddings(dataset):

    model = gensim.models.Word2Vec(min_count=3, epochs=10)
    model.train(dataset)
    return model


def extract_similarities_from_embeddings(model: gensim.models.Word2Vec,
                                         words: Iterable[str]):
    vocab_size = len(model.wv)
    similarties = np.array((len(words), vocab_size))
    for i, word in enumerate(words):
        closest_words = model.wv.most_similar(word, topn=None)
        similarties[i, :] = closest_words

    words = list(model.wv.index_to_key)
    res = pd.DataFrame(similarties, index=words)
    res.sort_index(inplace=True)
    return res


def extract_embeddings(model: gensim.models.Word2Vec):
    wv = model.wv
    vocab = wv.index_to_key()
    n_cols = wv.vector_size

    arr = np.zeros((vocab, n_cols), dtype=np.float32)
    for i, word in enumerate(vocab):
        vec = wv.get_vector(word)
        arr[i] = vec

    df = pd.DataFrame(arr, columns=range(n_cols), index=vocab)
    df.sort_index(inplace=True, ascending=True)

    return df


def main(args):
    dataset = load_dataset(args.dataset)
    model = learn_embeddings(dataset)
    embedding_df = extract_embeddings(model)
    seeds = load_seed(args.seed)

    embedding_df.to_csv(args.output)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out_similarities", required=True)
    parser.add_argument("--seed", required=True)
    parser.add_argument('--out_embeddings', reuqired=True)
    return parser.parse_args()


if __name__ == '__main__':
    pass
