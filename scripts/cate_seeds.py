"""
Author: Chi Zhang
Licence: MIT

This script is for replacing the out-of-vocabulary seeds 
with in-vocabulary seeds. Please refer README.md for detailed usages.
"""

import argparse as a

import pandas as pd
import logging
from rich.table import Table, Column
from rich.console import Console
from rich.logging import RichHandler
from utils.io import load_seed, load_vocab


def parse_args():
    parser = a.ArgumentParser()
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--out", required=True)
    parser.add_argument("--similarities", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--seeds", required=True)
    args = parser.parse_args()
    return args


def main(args):

    in_vocab, out_vocab = load_seed(args.seeds, False)
    vocab = load_vocab(args.vocab)
    similarities = pd.read_pickle(args.similarities)
    safe_vocab = set(vocab).intersection(set(similarities.index))
    similarities = similarities.loc[list(safe_vocab), :]
    replacements = find_in_vocab_replacements(out_vocab, similarities)
    visualize_replacement_seeds(out_vocab, replacements)
    logging.info(f"Saving seeds to {args.out}")
    with open(args.out, "w") as f:
        f.write("\n".join([
            *in_vocab,
            *replacements,
        ]))


def find_in_vocab_replacements(
    seeds: list,
    similarities: pd.DataFrame,
) -> list:
    """
    find the in vocabulary replacements for the out-of-vocabulary seeds

    Args:
        seeds (list[str]): A list of out-of-vocabulary seeds
        similarities(pd.Dataframe): DataFrame for all words of each topic
    Returns:
        A list of in-vocabulary replacements.
    """
    res = []
    for seed in seeds:
        sorted_col = similarities[seed].sort_values(ascending=False).head(3)
        in_vocab_seed = sorted_col.index[1]
        res.append(in_vocab_seed)
    return res


def visualize_replacement_seeds(old_seeds, new_seeds):
    """
        Visualize the replacements seeds
        old_seeds: A list of out-of-vocabular seeds
        new_seeds: A list of in-vocabulary replacements
    """
    console = Console()
    table = Table(
        Column("Out-vocab Seeds", style="cyan"),
        Column("Replaced Seeds", style="magenta"),
        show_header=True,
        header_style="bold magenta",
    )
    for old_seed, new_seed in zip(old_seeds, new_seeds):
        table.add_row(old_seed, new_seed)
    console.print(table)


if __name__ == "__main__":
    args = parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.INFO, handlers=[RichHandler()])
    else:
        logging.basicConfig(level=logging.WARNING, handlers=[RichHandler()])
    main(args)
