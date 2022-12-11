"""
Author: Chi Zhang
License: MIT

This script is used to run SeededLDA for a given dataset and seeds
"""
from lda.guidedlda import GuidedLDA
from utils.io import load_dataset, load_vocab, load_seed
from os.path import join as join_path
import numpy as np
import argparse
from sklearn.feature_extraction.text import CountVectorizer
import json
import os

import logging
from rich.logging import RichHandler


def dataset2matrix(dataset: list, vocab):
    """
    conut the frequency of each word in the vocabulary for a given dataset

    Args:
        dataset: a list of documents
        vocab: a list of words
    Returns:
        count of each word as a list, and the vocabulary
    """
    cv = CountVectorizer(ngram_range=(1, 1), vocabulary=vocab)
    X = cv.fit_transform(dataset)
    return X.toarray(), cv.vocabulary_


def main(args):
    # Intialize variables and loggers
    logger = logging.getLogger('guided_lda.py')
    DATA_ROOT = 'data'
    RESULTS_ROOT = 'results'
    dataset_name = args.dataset
    dataset_path = join_path(DATA_ROOT, dataset_name, 'corpus', 'corpus.txt')
    vocab_path = join_path(DATA_ROOT, dataset_name, 'vocab', 'vocab.pkl')
    seeds_path = join_path(DATA_ROOT, dataset_name, 'seeds.json')
    output_dir = join_path(RESULTS_ROOT, dataset_name, 'seeded_lda')

    # Create output directory
    logger.info(f"doing mkdir -p {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # Set output path
    output_path = join_path(output_dir, 'result.json')

    # Load dataset, vocab and seeds
    dataset = load_dataset(dataset_path)
    vocab = load_vocab(vocab_path)
    invocab_seeds, _ = load_seed(seeds_path, combine_result=False)
    logger.info(f"Using {invocab_seeds} as seeds for SeededLDA")

    seeds = [[seed] for seed in invocab_seeds]
    mat, word2id = dataset2matrix(dataset, vocab)

    seed_topics = {}
    # Run through guided LDA
    for t_id, st in enumerate(seeds):
        for word in st:
            seed_topics[word2id[word]] = t_id
    model = GuidedLDA(n_topics=len(seed_topics),
                      n_iter=100,
                      random_state=7,
                      refresh=20)

    model.fit(mat, seed_topics=seed_topics, seed_confidence=0.15)

    n_top_words = 10

    topic_word = model.topic_word_
    result = {}
    # Show results and save to file
    for i, topic_dist in enumerate(topic_word):
        topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words +
                                                                 1):-1]
        result[invocab_seeds[i]] = list(topic_words)
        logger.info(f"Topic ({invocab_seeds[i]}): {list(topic_words)}")
    logger.info(f"Saving Results to {output_path}")
    with open(output_path, 'w') as f:
        json.dump(result, f)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    logging.basicConfig(level=logging.INFO, handlers=[RichHandler()])
    main(args)
