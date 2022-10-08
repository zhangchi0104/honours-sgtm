import pandas as pd
import argparse
import numpy as np


def rank_reduction(embeddings: pd.DataFrame, similarities: pd.DataFrame,
                   similarity_threshold):
    seeds = similarities.columns
    labels = label_words(similarities, similarity_threshold)


def label_words(similarities: pd.DataFrame, similarity_threshold=0.7):
    vocab = similarities.index
    data = np.zeros((similarities.shape[0], 1))

    for i, word in enumerate(vocab):
        word_similarities = similarities.loc[word, :]
        max_similarity_idx = word_similarities.idxmax(axis=1)
        max_similarity = word_similarities.iloc[i, max_similarity_idx]
        data[
            i] = max_similarity_idx if max_similarity > similarity_threshold else -1
    res_df = pd.DataFrame(data, index=similarities.index)
    return res_df


def main(args):
    pass


def parse_args():
    pass


if __name__ == '__main__':
    pass