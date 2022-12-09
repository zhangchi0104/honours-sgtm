"""
Author: Chi Zhang
License: MIT

This script is used to evaluate the  results
"""

from pathlib import Path
import pandas as pd
import numpy as np
from argparse import ArgumentParser
from joblib import Parallel, delayed
import nltk
import json

from utils.io import load_vocab


def sum_pmi(words, mat, vocab):
    """
    Sums up the PMI of all word pairs in a word set

    Args: 
        words: a list of words
        mat: a co-occurence matrix
        vocab: the vocabulary

    Returns:
        the sum of PMI for the word set
    """
    res = 0
    word_count = sum(vocab.values())
    for i, word_a in enumerate(words):
        for word_b in words[i + 1:]:
            count_a = vocab.get(word_a, 0)
            count_b = vocab.get(word_b, 0)
            try:
                count_ab = mat.loc[word_a, word_b]
            except KeyError:
                count_ab = 0
            res += pmi(count_a, count_b, count_ab, word_count)
    return res


def pmi(
    a_count,
    b_count,
    ab_count,
    word_count,
):
    """
    Computes PMI for a given pair of words

    Args:
        a_count: the count of word a
        b_count: the count of word b
        ab_count: the count of word a and b co-occuring
        word_count: the total word count
    """
    corpus_word_count = float(word_count)
    prob_a = a_count / corpus_word_count
    prob_b = b_count / corpus_word_count
    prob_ab_cooccur = ab_count / corpus_word_count
    try:
        res = prob_ab_cooccur / (prob_a * prob_b)
    except ZeroDivisionError:
        res = 0
    return np.log(res) if res > 0 else 0


def npmi(
    a_count,
    b_count,
    ab_count,
    word_count,
):
    """
    Computes the normalized PMI for a given pair of words
    Args:
        a_count: the count of word a
        b_count: the count of word b
        ab_count: the count of word a and b co-occuring 
    """
    corpus_word_count = float(word_count)
    prob_ab = ab_count / corpus_word_count
    _pmi = pmi(a_count, b_count, ab_count, word_count)
    return _pmi / -np.log(prob_ab) if _pmi > 0 else 0


def sum_npmi(words, mat, vocab):
    """
    Sums up the NPMI of all word pairs in a word set
    Arfs:
        words: the word sets 
        mat: a co-occurence matrix
        vocab: the vocabulary 
    """
    res = 0
    word_count = sum(vocab.values())
    for i, word_a in enumerate(words):
        for word_b in words[i + 1:]:
            count_a = vocab.get(word_a, 0)
            count_b = vocab.get(word_b, 0)
            try:
                count_ab = mat.loc[word_a, word_b]
            except KeyError:
                count_ab = 0
            res += npmi(count_a, count_b, count_ab, word_count)
    return res


def compute_npmi_from_results(cooccur_mat, topics, word_sets, vocab):
    """
    wraps NPMI computatation with parallel job
    and prints results to stdout

    Args:
        cooccur_mat: the co-occurence matrix
        topics:  the list of topics
        word_sets: the word set for the topics 
    """

    def job(words, cooccur_mat, topic):
        res = sum_npmi(words, cooccur_mat, vocab)
        print(f'\t{topic}:{res}')
        return res

    print("====NPMIS====")
    _npmi = Parallel(n_jobs=min(20, len(topics)), prefer='threads')(
        delayed(job)(word_sets[topics[i]], cooccur_mat, topics[i])
        for i in range(len(topics)))
    res = np.sum(_npmi) / len(_npmi)
    print(f"\t<SUM>: {res}")
    return res


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("files", nargs="+", help="input files")
    parser.add_argument(
        "--out",
        "-o",
        help="path to output file, Default: ./evaluations.json",
        default=Path('./evaluations.json'),
        type=Path)
    parser.add_argument("--cooccur",
                        "-c",
                        required=True,
                        type=str,
                        help="path to precomputed co-occurence matrix")
    parser.add_argument("--vocab",
                        "-v",
                        required=True,
                        type=str,
                        help="path to vocabulary")
    return parser.parse_args()


def evaluation(files, cooccur_mat, vocab, out, n_words=5):
    print("SUMMARY")
    print("=" * 80)
    obj = {}
    for file in files:
        print(file)
        scores = {}
        basename = file.split('/')[-1].split('.')[0]
        if file.endswith('.csv') or file.endswith('.pkl'):
            scores_df = pd.read_pickle(file)
            # construct wordset
            word_sets = build_word_sets(scores_df, n_words=n_words)

        else:
            f = open(file, 'r')
            word_sets = json.load(f)
            f.close()
        topics = list(word_sets.keys())
        # compute PMI
        _pmi = compute_pmi_from_results(cooccur_mat, topics, word_sets, vocab)
        _npmi = compute_npmi_from_results(cooccur_mat, topics, word_sets,
                                          vocab)
        _distinctivess = compute_distinctiveness(word_sets)
        scores['distinctiveness'] = _distinctivess
        scores['pmi'] = _pmi
        scores['npmi'] = _npmi
        scores['word_set'] = word_sets

        obj[basename] = scores

    with open(out, 'w') as f:
        json.dump(obj, f)


def compute_distinctiveness(word_sets):
    """
    Computes the distinctiveness of a word set 
    Args:
        word_sets ([list[list[str]]]): the word sets
    """

    # Some initializations
    words = list(word_sets.values())
    topics = list(word_sets.keys())
    words = [set(v) for v in words]
    total_items = 0
    unique_words = set()
    # Do Union of all words
    for s, _ in zip(words, topics):
        total_items += len(s)
        unique_words = unique_words.union(s)

    distinctiveness = len(unique_words) / total_items

    print("====DISTINCTIVENESS====")
    print(f"\t<SUM>: {distinctiveness}")
    return distinctiveness


def compute_pmi_from_results(cooccur_mat, topics, word_sets, vocab):
    """
    Comptuse PMI in parallel
    Args:
        cooccur_mat: the co-occurence matrix
        topics:  the list of topics
        word_sets: the word set for the topics
        vocab: the vocabulary
    """
    _pmi = []
    print("====PMIS====")

    def job(words, cooccur_mat, topic):
        res = sum_pmi(words, cooccur_mat, vocab)
        print(f'\t{topic}:{res}')
        return res

    _pmi = Parallel(n_jobs=min(20, len(topics)), prefer='threads')(
        delayed(job)(word_sets[topics[i]], cooccur_mat, topics[i])
        for i in range(len(topics)))
    res = np.sum(_pmi) / len(_pmi)
    print(f"\t<SUM>: {res}")
    return res


def build_word_sets(scores_df, n_words=5):
    """
    Choose top n words from the scores dataframe
    Args:
        scores_df (pd.DataFrame): the dataframe for the combined scores
        n_words (int, optional): the number of words to choose. Defaults to 5. 
    """
    word_sets = {}
    for topic in scores_df.columns:
        word_sets[topic] = list(
            scores_df[topic].sort_values(ascending=False).head(n_words).index)
        if topic not in word_sets[topic]:
            word_sets[topic].pop()
            word_sets[topic].append(topic)
    print("====WORD SET====")
    for topic, item in word_sets.items():
        print(f"\t{topic}:{item}")
    return word_sets


def main():
    args = parse_args()
    cooccur_mat = pd.read_csv(args.cooccur, index_col=0)
    vocab = load_vocab(args.vocab, False)
    evaluation(args.files, cooccur_mat, vocab, args.out)


if __name__ == '__main__':
    main()
