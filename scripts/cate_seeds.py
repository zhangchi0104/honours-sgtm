import argparse as a
from array import array
from shutil import move
from typing import Iterable

import pandas as pd
import logging
from rich.table import Table, Column
from rich.console import Console
from rich.logging import RichHandler
import numpy as np
from utils.io import load_seed, load_vocab
import gensim


def parse_args():
    parser = a.ArgumentParser()
    parser.add_argument("--verbose", "-v", action="store_true")
    sub_parser = parser.add_subparsers(dest='command')
    seed_parser = sub_parser.add_parser("gen_seeds")
    seed_parser.add_argument("--out", required=True)
    seed_parser.add_argument("--similarities", required=True)
    seed_parser.add_argument("--vocab", required=True)
    seed_parser.add_argument("--seeds", required=True)
    args = parser.parse_args()
    return args


def main(args):
    if args.command == "gen_seeds":
        in_vocab, out_vocab = load_seed(args.seeds, False)
        vocab = load_vocab(args.vocab)
        similarities = pd.read_csv(args.similarities, index_col=0)
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


def find_in_vocab_replacements(seeds: list, similarities: pd.DataFrame):
    res = []
    for seed in seeds:
        sorted_col = similarities[seed].sort_values(ascending=False).head(3)
        in_vocab_seed = sorted_col.index[1]
        res.append(in_vocab_seed)
    return res


def visualize_replacement_seeds(old_seeds, new_seeds):
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
