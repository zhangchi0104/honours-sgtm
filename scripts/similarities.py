from random import seed
import pandas as pd
import numpy as np
import logging
from rich.logging import RichHandler
import argparse
from utils import embeddings
from utils.embeddings import cosine_similarity_with_topic
from utils.visualize import visualize_results
from utils.io import load_model, load_tokenizer, load_vocab, load_seed
from utils.embeddings import batch_cosine_similarity

logging.basicConfig(level=logging.INFO,
                    handlers=[RichHandler()],
                    format="%(message)s")


def parse_args():
    parser = argparse.ArgumentParser()
    sub_parser = parser.add_subparsers(dest="command")
    bert_parser = sub_parser.add_parser("bert")

    parser.add_argument("--vocab", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--out_embeddings", type=str)
    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("--seeds", type=str, required=True)
    # Bert Parser
    bert_parser.add_argument("--weights", type=str, default="")
    bert_parser.add_argument("--tokenizer", type=str, default="")
    # CatE Parser
    cate_parser = sub_parser.add_parser("cate")
    cate_parser.add_argument("--topic", type=str, required=True)
    cate_parser.add_argument("--words", type=str, required=True)

    cate_parser.add_argument("--similarities", type=str, required=True)
    args = parser.parse_args()
    if args.command == "bert" and args.tokenizer.strip(
    ) != "" and args.weights.strip() == "":
        raise argparse.ArgumentError(
            args.tokenizer,
            "Cannot user pretrained model when using custom tokenizer",
        )
    return args


def main(args):
    # load datasets
    if args.command == "bert":
        df = bert_embeddings(args)
    else:
        df = cate_embeddings(args)
    logging.info(f"Saving embeddings to {args.output}")
    df.to_csv(args.output)
    visualize_results(df)


def bert_embeddings(args):
    seeds = load_seed(args.seeds)
    vocab_set = load_vocab(args.vocab)
    df = pd.DataFrame(None,
                      index=list(vocab_set),
                      columns=seeds,
                      dtype=np.float32)
    data = np.zeros((len(vocab_set), len(seeds)))
    logging.info(
        f"Created a dataframe for cosine similarities with shape {df.shape}")
    logging.info(f"Using device {args.device}")
    model = load_model(args.weights, device=args.device)
    model.to(args.device)
    tokenizer = load_tokenizer(args.tokenizer)
    for i, topic in enumerate(df.columns):
        similarities, embeddings = cosine_similarity_with_topic(
            topic,
            vocab_set,
            tokenizer,
            model,
            batch_size=8192,
            device=args.device)
        data[:, i] = similarities

    df = pd.DataFrame(data, index=vocab_set, columns=seeds)
    embeddings_df = pd.DataFrame(data, index=vocab_set, columns=range(768))
    if args.out_embeddings:
        logging.info(
            f"writing embeddings with shape {embeddings_df.shape} to {args.out_embeddings}"
        )
        embeddings_df.to_csv(args.out_embeddings)
    return df


def cate_embeddings(args):
    topics = load_seed(args.seeds)
    topic_embeddings = load_cate_embeddings(args.topic)
    vocab_embeddings = load_cate_embeddings(args.words)

    res = np.zeros((vocab_embeddings.shape[0], len(topics)))
    for i, topic in enumerate(topic_embeddings.index):
        similarities = batch_cosine_similarity(topic_embeddings.loc[topic],
                                               vocab_embeddings.to_numpy())
        res[:, i] = similarities
    df = pd.DataFrame(res, index=vocab_embeddings.index, columns=topics)
    logging.info(
        f"Created a dataframe for cosine similarities with shape {df.shape}")
    return df


def load_cate_embeddings(path):
    df = pd.read_csv(path, index_col=0, skiprows=[0], sep=" ", header=None)
    logging.info(f"Loaded embeddings with shape {df.shape} from {path}")
    return df.dropna(axis=1, how="all")


if __name__ == "__main__":
    args = parse_args()
    main(args)