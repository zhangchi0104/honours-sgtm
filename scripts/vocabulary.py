import argparse
import pandas as pd
import pickle
import logging
from rich.logging import RichHandler
from rich.progress import track

logging.basicConfig(handlers=[RichHandler()], level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input file", type=str)

    parser.add_argument("--min_count",
                        help="Minimum number of occurrences",
                        type=int,
                        default=3)

    parser.add_argument("--cooccur_path",
                        "-c",
                        help="Path to co-occurrence matrix",
                        type=str,
                        default=None)
    parser.add_argument("--vocabulary_path",
                        "-v",
                        help="Path to vocabulary",
                        type=str,
                        required=True)

    return parser.parse_args()


def main(args):
    args = parse_args()
    lines = None
    with open(args.input, 'r') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines if line != '']
    vocab = build_vocabulary(lines, args.min_count)
    vocab = [word.strip() for word in vocab]
    vocab = set(vocab)
    with open(args.vocabulary_path, 'wb') as f:
        logging.info("Saving vocabulary to %s", args.vocabulary_path)
        pickle.dump(vocab, f)
        logging.info("Done saving vocabulary")
    if args.cooccurrence_matrix_path is not None:
        coocurrence_matrix = build_cooccurance_matrix(lines, vocab)
        coocurrence_matrix.to_csv(args.cooccurrence_matrix_path)


def build_vocabulary(lines, min_count):
    all_occurences = {}
    loop = track(lines, description="Building vocabulary")
    for line in loop:
        words = line.split(' ')
        for word in words:
            if word in all_occurences.keys():
                all_occurences[word] += 1
            else:
                all_occurences[word] = 1

    return {
        word
        for word, count in all_occurences.items() if count >= min_count
    }


def build_cooccurance_matrix(
    lines,
    vocabulary,
):
    logging.info("Starting building co-occurrence matrix")
    coocurrence_matrix = pd.DataFrame(index=vocabulary, columns=vocabulary)
    coocurrence_matrix.fillna(0, inplace=True)
    for line in track(lines, desc="Building co-occurrence matrix"):
        words = line.split(' ')
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                if words[i] in vocabulary and words[j] in vocabulary:
                    coocurrence_matrix.loc[words[i], words[j]] += 1
                    coocurrence_matrix.loc[words[j], words[i]] += 1
    for i in range(len(vocabulary)):
        coocurrence_matrix.iloc[i, i] = 0
    return coocurrence_matrix


if __name__ == "__main__":
    args = parse_args()
    main(args)