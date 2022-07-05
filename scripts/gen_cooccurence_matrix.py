import pickle
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab",
                        "-v",
                        help="path to vocabulary file",
                        required=True)
    parser.add_argument(
        "dataset",
        nargs=1,
        help="path to input dataset to generate the co-occurence matrix")
    parser.add_argument("--out",
                        "-o",
                        default="./data/vocab/cooccurence_matrix-agnews.csv")
    return parser.parse_args()


def generate_cooccurence_matrix(vocab_path, dataset_path):
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    data = np.zeros((len(vocab), len(vocab)))
    cooccur = pd.DataFrame(columns=vocab.keys(), index=vocab.keys(), data=data)
    f = open(dataset_path, 'r')
    lines = f.readlines()
    f.close()
    for line in tqdm(lines):
        words = [word for word in line.split(' ') if word in vocab.keys()]
        for word_x in words:
            for word_y in words:
                if word_x != word_y:
                    cooccur.loc[word_x, word_y] += 1
    return cooccur


def main():
    args = parse_args()
    mat = generate_cooccurence_matrix(args.vocab, args.dataset[0])
    mat.to_csv(args.out)


if __name__ == '__main__':
    main()