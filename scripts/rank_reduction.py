"""
Author: Chi Zhang
License: MIT

Performs rank reduction on the embeddings with UMAP
"""
import pandas as pd
import argparse
import numpy as np
from rich.logging import RichHandler
import logging
import umap


def rank_reduction(embeddings: pd.DataFrame, similarities: pd.DataFrame,
                   similarity_threshold, n_dim):
    """
    Preforms rank reduction on the embeddings with UMAP

    Args:
        embeddings: a dataframe of embeddings
        similarities: a dataframe of cosine similarities
        similarity_threshold: a threshold for the cosine similarities 
        n_dim: the number of dimensions to reduce to 
    Returns:
        a dataframe of reduced embeddings
    """

    # Label the words with corresponding topics
    labels = label_words(similarities, similarity_threshold)
    fitter = umap.UMAP(n_components=n_dim).fit(embeddings.to_numpy(),
                                               y=labels.to_numpy())
    # Put the embeddings into the dataset
    reduced_embeddings = fitter.embedding_
    reduced_embeddings_df = pd.DataFrame(reduced_embeddings,
                                         index=embeddings.index)

    return reduced_embeddings_df


def label_words(similarities: pd.DataFrame, similarity_threshold=0.8):
    """
    label the words with corresponding topics, if the cosine similarity is
    larger than the threshold, then label the word with the topic, otherwise
    label the word with -1

    Args:
        similarities: a dataframe of cosine similarities
        similarity_threshold: a threshold for the cosine similarities
    Returns:
        a dataframe of labels
    """
    vocab = similarities.index
    data = np.zeros((similarities.shape[0], 1))
    topics = list(similarities.columns)

    for i, word in enumerate(vocab):
        word_similarities = similarities.loc[word, :]
        max_similarity_idx = word_similarities.idxmax(axis=0)
        max_similarity = word_similarities.loc[max_similarity_idx]
        if max_similarity > similarity_threshold:
            data[i] = topics.index(max_similarity_idx)
        else:
            data[i] = -1
    res_df = pd.DataFrame(data, index=similarities.index, columns=['label'])
    return res_df


def main(args):
    similarities = pd.read_pickle(args.similarities)
    logging.info(
        f"Loaded cosine similarities from {args.similarities} with shape {similarities.shape}"
    )
    embeddings = pd.read_pickle(args.embeddings)
    logging.info(
        f"Loaded embeddings from {args.embeddings} with shape {embeddings.shape}"
    )
    reduced_embeddings = rank_reduction(embeddings, similarities,
                                        args.similarity_threshold, args.ndim)
    logging.info(f"Writing results to {args.output}")
    reduced_embeddings.to_pickle(args.output)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--similarities", "-s", type=str, required=True)
    parser.add_argument("--embeddings", "-e", type=str, required=True)
    parser.add_argument("--ndim", '-n', type=int, default=80)
    parser.add_argument("--similarity_threshold",
                        "-t",
                        type=float,
                        default=0.7)
    parser.add_argument("--output", "-o", type=str, required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    logging.basicConfig(level=logging.INFO, handlers=[RichHandler()])
    main(args)