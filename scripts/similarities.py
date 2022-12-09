import json
import pandas as pd
import numpy as np
import logging
from rich.logging import RichHandler
import argparse
import torch
from utils.embeddings import cosine_similarity_with_topic
from utils.visualize import visualize_results
from utils.io import load_model, load_tokenizer, load_vocab, load_seed
from utils.embeddings import batch_cosine_similarity
from tqdm.auto import tqdm

logging.basicConfig(level=logging.INFO,
                    handlers=[RichHandler()],
                    format="%(message)s")


def parse_args():
    parser = argparse.ArgumentParser()
    sub_parser = parser.add_subparsers(dest="command")
    bert_parser = sub_parser.add_parser("bert")

    parser.add_argument("--vocab",
                        type=str,
                        required=True,
                        help="Path to vocabulary")
    parser.add_argument("--output",
                        type=str,
                        required=True,
                        help="The path to output similarities")
    parser.add_argument("--out_embeddings",
                        type=str,
                        help="If set, save embeddings to specified path")

    parser.add_argument("--seeds",
                        type=str,
                        required=True,
                        help="Path to seeds.json")
    parser.add_argument(
        "--embeddings",
        type=str,
        default=None,
        help=
        "If specified, use embeddings from the path, rather than extracting them from the model"
    )
    parser.add_argument("--out_words",
                        type=str,
                        default=None,
                        help="If set, save top 10 words to specified path")
    # Bert Parser
    bert_parser.add_argument("--weights",
                             type=str,
                             help="Path to model weights")
    bert_parser.add_argument("--tokenizer",
                             type=str,
                             help="Path to the tokenizer")
    bert_parser.add_argument(
        "--device",
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help="device to use for running the model, either 'cuda' or 'cpu'")
    # CatE Parser
    cate_parser = sub_parser.add_parser("cate", help="CatE embeddings")
    cate_parser.add_argument("--topic",
                             type=str,
                             required=True,
                             help="Path to CaTE topic embeddings")
    cate_parser.add_argument("--words",
                             type=str,
                             required=True,
                             help="Path to CaTE word embeddings ")

    args = parser.parse_args()
    if args.command == "bert":
        has_weights = args.weights is not None
        has_tokenizer = args.tokenizer is not None
        if not has_tokenizer == has_weights:
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
    df.index = df.index.astype(str)
    df.to_pickle(args.output)
    words_set = {}
    if args.out_words:
        for topic in df.columns:
            col = df[topic].sort_values(ascending=False)
            words_set[topic] = col.head(10).index.to_list()
        logging.info(f"Writing word sets to {args.out_words}")
        with open(args.out_words, 'w') as f:
            json.dump(words_set, f)
    visualize_results(df)


def bert_embeddings(args):
    """
    Extracts embeddings from the model and computes cosine similarities
    """

    # Initialization
    seeds = load_seed(args.seeds)
    vocab_set = load_vocab(args.vocab)
    all_vocab = set(vocab_set).union(set(seeds))
    all_vocab = sorted(list(all_vocab))
    df = pd.DataFrame(None,
                      index=list(all_vocab),
                      columns=seeds,
                      dtype=np.float32)
    data = np.zeros((len(all_vocab), len(seeds)))
    logging.info(
        f"Created a dataframe for cosine similarities with shape {df.shape}")
    logging.info(f"Using device {args.device}")
    model = load_model(args.weights, device=args.device)
    tokenizer = load_tokenizer(args.tokenizer)
    if not args.embeddings:
        # Extract embeddings from model and compute similarities
        for i, topic in enumerate(df.columns):
            similarities = cosine_similarity_with_topic(topic,
                                                        all_vocab,
                                                        tokenizer,
                                                        model,
                                                        batch_size=3072,
                                                        device=args.device)
            data[:, i] = similarities
            df = pd.DataFrame(data, index=all_vocab, columns=seeds)
    else:
        # Load embeddings from file and compute similarities
        word_embeddings = pd.read_pickle(args.embeddings)
        for i, topic in enumerate(df.columns):
            topic_embedding = word_embeddings.loc[topic, :]
            similarities = batch_cosine_similarity(topic_embedding.to_numpy(),
                                                   word_embeddings.to_numpy())
            data[:, i] = similarities
        df = pd.DataFrame(data, index=word_embeddings.index, columns=seeds)
    if args.out_embeddings:
        # Allocate memory
        embeddings_arr = np.zeros((len(all_vocab), 768))
        batch_size = 3072
        embeddings_df = pd.DataFrame(embeddings_arr,
                                     index=all_vocab,
                                     columns=range(768))
        # Extract embeddings from model
        for lo in tqdm(range(0, len(all_vocab), batch_size)):
            hi = min(lo + batch_size, len(all_vocab))
            batch = all_vocab[lo:hi]
            inputs = tokenizer(batch, return_tensors='pt', padding=True)
            inputs.to(args.device)
            with torch.no_grad():
                outputs = model(**inputs)
                embeddings = outputs.last_hidden_state.detach().cpu().numpy(
                )[:, 1, :]
                embeddings_df.loc[batch, :] = embeddings

        logging.info(
            f"writing embeddings with shape {embeddings_df.shape} to {args.out_embeddings}"
        )
        # Save embeddings to file
        embeddings_df.to_pickle(args.out_embeddings)
    return df


def cate_embeddings(args):
    """
    Computes similarities for CatE embeddings
    """
    df = None
    if not args.embeddings:
        # Computes similarities for raw embeddings from file

        # Load embeddings and seeds from file
        topics = load_seed(args.seeds)
        topic_embeddings = load_cate_embeddings(args.topic)
        vocab_embeddings = load_cate_embeddings(args.words)

        # Computes similarities for each topic
        res = np.zeros((vocab_embeddings.shape[0], len(topics)))
        for i, topic in enumerate(topic_embeddings.index):
            similarities = batch_cosine_similarity(topic_embeddings.loc[topic],
                                                   vocab_embeddings.to_numpy())
            res[:, i] = similarities
        # Put the data into the dataframe
        df = pd.DataFrame(res, index=vocab_embeddings.index, columns=topics)
        if (args.out_embeddings):
            for topic in topic_embeddings.index:
                vocab_embeddings.loc[topic, :] = topic_embeddings.loc[topic, :]
                vocab_embeddings.to_pickle(args.out_embeddings)
        logging.info(
            f"Created a dataframe for cosine similarities with shape {df.shape}"
        )
    else:
        # Computes similarities by using embeddings from the given file
        topic_embeddings = load_cate_embeddings(args.topic)
        embeddings = pd.read_pickle(args.embeddings)
        topics = load_seed(args.seeds)
        res = np.zeros((embeddings.shape[0], len(topics)))
        for i, topic in enumerate(topic_embeddings.index):
            similarities = batch_cosine_similarity(embeddings.loc[topic],
                                                   embeddings.to_numpy())
            res[:, i] = similarities
        df = pd.DataFrame(res, index=embeddings.index, columns=topics)
    return df


def load_cate_embeddings(path):
    df = pd.read_csv(path, index_col=0, skiprows=[0], sep=" ", header=None)
    logging.info(f"Loaded embeddings with shape {df.shape} from {path}")
    return df.dropna(axis=1, how="all")


if __name__ == "__main__":
    args = parse_args()
    main(args)