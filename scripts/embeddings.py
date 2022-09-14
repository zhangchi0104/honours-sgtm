from ast import arg
from gc import set_debug
from pydoc_data.topics import topics
import pandas as pd
import numpy as np
import pickle
import logging
from rich.logging import RichHandler
import argparse
from transformers import BertModel, BertTokenizer
import torch
from utils.embeddings import cosine_similarity_with_topic
from rich.console import Console
from rich.table import Table, Column

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


def visualize_results(df: pd.DataFrame):
    console = Console()
    table = Table("Topic", "1st", "2nd", "3rd", "4th", "5th")
    for topic in df.columns:
        col = df[topic].sort_values(ascending=False)
        words = col.head(5).index.to_list()
        table.add_row(topic, *words)
    console.print(table)


def load_vocab(path):
    vocab_set = None
    with open(path, 'rb') as f:
        vocab_set = pickle.load(f)

    vocab = [word.strip() for word in vocab_set]
    vocab = list(set(vocab))
    logging.info(f"Loaded {len(vocab)} words from {path}")
    return vocab


def load_seed(path):
    seeds = []
    with open(path, 'r') as f:
        seeds = f.readlines()
    seeds = [seed.strip() for seed in seeds]
    logging.info(f"loaded {len(seeds)} seeds from {path}")
    return seeds


def load_model(path):
    if path is None or path.strip() == '':
        logging.info("loading pretrained model")
        return BertModel.from_pretrained('bert-base-uncased')
    logging.info(f"loading BERT model from {path}")
    model = BertModel.from_pretrained('bert-base-uncased')
    model.load_state_dict(torch.load(path), strict=False)
    return model.eval()


def load_tokenizer(path):
    if path is None or path.strip() == '':
        logging.info(f'loaded pretrained tokenizer')
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        return tokenizer
    logging.info(f"loading pretrained tokenizer from {path}")
    tokenizer = BertTokenizer(path)
    return tokenizer


if __name__ == "__main__":
    args = parse_args()
    main(args)