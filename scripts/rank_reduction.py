from functools import reduce
import pandas as pd
import argparse
import numpy as np
from rich.logging import RichHandler
from utils.visualize import visualize_results
import logging
import umap


def rank_reduction(embeddings: pd.DataFrame, similarities: pd.DataFrame,
                   similarity_threshold):
    labels = label_words(similarities, similarity_threshold)
    fitter = umap.UMAP(n_components=50).fit(embeddings.to_numpy(),
                                            y=labels.to_numpy())
    reduced_embeddings = fitter.embedding_
    reduced_embeddings_df = pd.DataFrame(reduced_embeddings,
                                         index=embeddings.index)

    return reduced_embeddings_df


def label_words(similarities: pd.DataFrame, similarity_threshold=0.8):
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
    res_df.groupby(['label'])
    print(res_df)
    return res_df


def main(args):
    similarities = pd.read_csv(args.similarities, index_col=0)
    logging.info(
        f"Loaded cosine similarities from {args.similarities} with shape {similarities.shape}"
    )
    embeddings = pd.read_pickle(args.embeddings)
    logging.info(
        f"Loaded embeddings from {args.embeddings} with shape {embeddings.shape}"
    )
    reduced_embeddings = rank_reduction(embeddings, similarities,
                                        args.similarity_threshold)
    logging.info(f"Writing results to {args.output}")
    reduced_embeddings.to_pickle(args.output)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--similarities", type=str, required=True)
    parser.add_argument("--embeddings", type=str, required=True)
    parser.add_argument("--similarity_threshold", type=float, default=0.7)
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    logging.basicConfig(level=logging.INFO, handlers=[RichHandler()])
    main(args)