import pandas as pd
import numpy as np
import logging
from rich.logging import RichHandler
import argparse
from utils.embeddings import cosine_similarity_with_topic
from utils.visualize import visualize_results
from utils.io import load_model, load_tokenizer, load_vocab, load_seed

logging.basicConfig(level=logging.INFO,
                    handlers=[RichHandler()],
                    format="%(message)s")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab", type=str, required=True)  # dataset.txt
    parser.add_argument("--seeds", type=str, required=True)
    parser.add_argument("--weights", type=str, default="")
    parser.add_argument("--tokenizer", type=str, default="")
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    if args.tokenizer.strip() != "" and args.weights.strip() == "":
        raise argparse.ArgumentError(
            args.tokenizer,
            "Cannot user pretrained model when using custom tokenizer",
        )
    return args


def main(args):
    # load datasets
    seeds = load_seed(args.seeds)
    vocab_set = load_vocab(args.vocab)
    df = pd.DataFrame(None,
                      index=list(vocab_set),
                      columns=seeds,
                      dtype=np.float32)
    data = np.zeros((len(vocab_set), 20))

    logging.info(
        f"Created a dataframe for cosine similarities with shape {df.shape}")
    model = load_model(args.weights)
    tokenizer = load_tokenizer(args.tokenizer)
    for i, topic in enumerate(df.columns):
        similarities = cosine_similarity_with_topic(topic,
                                                    vocab_set,
                                                    tokenizer,
                                                    model,
                                                    batch_size=512)
        data[:, i] = similarities
    logging.info(f"Saving embeddings to {args.output}")
    df = pd.DataFrame(data, index=vocab_set, columns=seeds)
    df.to_csv(args.output)
    visualize_results(df)


if __name__ == "__main__":
    args = parse_args()
    main(args)