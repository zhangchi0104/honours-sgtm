from email.mime import base
from pathlib import Path
import pandas as pd
import numpy as np
from argparse import ArgumentParser

import pickle


def prob_word(word, mat):
    return np.sum(mat[word]) / np.sum(mat.to_numpy())


def prob_cooccur(a, b, mat):
    return mat.loc[a, b] / np.sum(mat.to_numpy())


def sum_pmi(words, mat):
    res = 0
    for i, word_a in enumerate(words):
        for word_b in words[i + 1:]:
            res += pmi(word_a, word_b, mat)
    return res


def pmi(a, b, mat):
    prob_a = prob_word(a, mat)
    prob_b = prob_word(b, mat)
    prob_ab_cooccur = prob_cooccur(a, b, mat)
    res = prob_ab_cooccur / (prob_a * prob_b)
    return np.log(res) if res > 0 else 0

def nmpi(a, b, mat):
    return pmi(a, b, mat) / -np.log(prob_cooccur(a, b))

def sum_npmi(words, mat):
    res = 0
    for i, word_a in enumerate(words):
        for word_b in words[i + 1:]:
            res += npmi(word_a, word_b, mat)
    return res
    pass

def compute_npmi_from_results(cooccur_mat, topics, word_sets):
    npmi = []
    print("====NPMIS====")
    for topic in topics:
        val = sum_npmi(word_sets[topic], cooccur_mat)
        pmi.append(val)
        print(f'\t{topic}:{val}')
    return np.sum(npmi)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("files", nargs="+", help="input files")
    parser.add_argument("--out",
                        "-o",
                        default=Path('./results/pmi/evaluations.pkl'),
                        type=Path)
    parser.add_argument("--cooccur",
                        "-c",
                        default="data/vocab/cooccurence_matrix-agnews.csv",
                        type=str,
                        help="path to precomputed co-occurence matrix")

    return parser.parse_args()


def evaluation(files, cooccur_mat, out):
    print("SUMMARY")
    print("=" * 80)
    obj = {}
    for file in files:
        print(file)
        scores = {}
        basename = file.split('/')[-1].split('.')[0]
        scores_df = pd.read_csv(file, index_col='_vocab')
        # construct wordset
        word_sets = build_word_sets(scores_df)
        # compute PMI
        pmi = compute_pmi_from_results(cooccur_mat, scores_df.columns, word_sets)
        scores['pmi'] = pmi
        scores['word_set'] = word_sets
        print(f"\t<SUM>: {scores['pmi']}")
        obj[basename] = scores

    with open(out, 'wb') as f:
        pickle.dump(obj, f)


def compute_pmi_from_results(cooccur_mat, topics, word_sets):
    pmi = []
    print("====PMIS====")
    for topic in topics:
        val = sum_pmi(word_sets[topic], cooccur_mat)
        pmi.append(val)
        print(f'\t{topic}:{val}')
    return np.sum(pmi)


def build_word_sets(scores_df, n_words=5):
    word_sets = {}
    for topic in scores_df.columns:
        word_sets[topic] = list(
            scores_df[topic].sort_values(ascending=False).head(n_words).index)
    print("====WORD SET====")
    for topic, item in word_sets.items():
        print(f"\t{topic}:{item}")
    return word_sets


def main():
    args = parse_args()
    cooccur_mat = pd.read_csv(args.cooccur, index_col=0)
    evaluation(args.files, cooccur_mat, args.out)


if __name__ == '__main__':
    main()