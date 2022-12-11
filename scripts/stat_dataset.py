"""
Author: Chi Zhang
License: MIT

count the number of words and phrases in a dataset
A phrase is defined by a sequence of words joined by underscore "_" 
"""
import argparse as a
from rich.table import Table
from rich.console import Console


def parse_args():
    parser = a.ArgumentParser()
    parser.add_argument('datasets', nargs="+")

    return parser.parse_args()


def main(args):
    console = Console()
    datasets = args.datasets
    table = Table("Dataset", "# Words", "# Phrases", "Total")
    for dataset in datasets:
        n_words, n_phrases = stat_dataset(dataset)
        total = n_words + n_phrases
        table.add_row(
            dataset,
            f"{format(n_words, ',')}",
            f"{format(n_phrases, ',')}",
            f"{format(total, ',')}",
        )
    console.print(table)


def stat_dataset(dataset_name):
    """
    Count the number of words and phrases for a given dataset name

    Args:
        dataset_name (str): name of the dataset
    """
    dataset_path = f'./data/{dataset_name}/corpus/corpus.txt'
    return stat_dataset_by_path(dataset_path)


def stat_dataset_by_path(dataset_path):
    """
    Count the number of words and phrases in a dataset

    Args:
        dataset_path (str): path to the dataset file
    Returns:
        (int, int): (number of words, number of phrases) 
    """
    f = open(dataset_path, 'r')
    lines = f.readlines()
    n_words = 0
    n_phrases = 0
    for line in lines:
        parts = line.strip().split(' ')
        total = len(parts)
        phrase_count = len([part for part in parts if '_' in part])
        word_count = total - phrase_count
        n_words += word_count
        n_phrases += phrase_count

    return n_words, n_phrases


if __name__ == '__main__':
    args = parse_args()
    main(args)
