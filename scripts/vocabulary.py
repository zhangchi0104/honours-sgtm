import argparse
import pandas as pd
import pickle
import logging
from rich.logging import RichHandler
from rich.progress import track
from tqdm import tqdm

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

    if isinstance(vocab, (list, set)):
        vocab = [word.strip() for word in vocab]
        vocab = set(vocab)
    logging.info(f"Vocabulary size: {len(vocab)}")
    with open(args.vocabulary_path, 'wb') as f:
        logging.info("Saving vocabulary to %s", args.vocabulary_path)
        pickle.dump(vocab, f)
        logging.info("Done saving vocabulary")
    # return
    if args.cooccur_path is not None:
        vocab = list(vocab)
        coocurrence_matrix = build_cooccurance_matrix(lines, vocab)
        logging.info(f"Saving dataframe to {args.cooccur_path}")
        coocurrence_matrix.to_csv(args.cooccur_path)


def build_vocabulary(lines, min_count):
    all_occurences = {}
    # Do the counting
    loop = track(lines, description="Building vocabulary")
    for line in loop:
        words = line.split(' ')
        for word in words:
            if word in all_occurences.keys():
                all_occurences[word] += 1
            else:
                all_occurences[word] = 1

    return {
        word: count
        for word, count in all_occurences.items()
        if count >= min_count and not word.isspace()
    }


def build_cooccurance_matrix(
    lines,
    vocabulary,
):
    """
    Build a co-occurrence matrix from a list of lines 
    """
    from sklearn.feature_extraction.text import CountVectorizer

    # Do unigram counting
    logging.info("making unigram matrix")
    cv = CountVectorizer(ngram_range=(1, 1), vocabulary=vocabulary)
    x = cv.fit_transform(lines)
    logging.info("Computing occurrence matrix")
    # Compute co-occurrence matrix using linear algebra
    co_x = x.T * x
    co_x.setdiag(0)
    data = co_x.toarray()
    names = cv.get_feature_names_out()
    logging.info(
        "Starting building dataframe, it may take a while depending on the size of the vocabulary"
    )
    df = pd.DataFrame(data, columns=names, index=names)
    return df


if __name__ == "__main__":
    args = parse_args()
    main(args)
