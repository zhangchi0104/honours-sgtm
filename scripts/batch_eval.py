"""
Author: Chi Zhang
Licence: MIT

This script is for evaluating the results on different ensemble parameters settings. 
Please refer README.md for detailed usages.
"""
import argparse
from pathlib import Path
from ensemble_ranking import run_ensemble_rankings
from evaluation import evaluation
import pandas as pd
import datetime
import sys
import logging
from rich.logging import RichHandler
from utils.io import load_vocab
import os


def main(args):
    _dataset = args.dataset
    # sets the similarities file
    similarity_name = 'similarities.pkl' if not args.reduced_similarities else 'similarities_reduced.pkl'
    # ROOT folder of the project, default to $PWD
    PROJECT_ROOT = Path(os.getcwd()).expanduser().absolute()
    CONFIG = {
        # local_weight, global_weight, rho
        "specs": [
            # changes in weights
            (0.1, 0.9, 0.5),
            (0.2, 0.8, 0.5),
            (0.3, 0.7, 0.5),
            (0.4, 0.6, 0.5),
            (0.5, 0.5, 0.5),
            (0.6, 0.4, 0.5),
            (0.7, 0.3, 0.5),
            (0.8, 0.2, 0.5),
            (0.9, 0.1, 0.5),
            # changes in rho
            (0.5, 0.5, 0.1),
            (0.5, 0.5, 0.3),
            (0.5, 0.5, 0.7),
            (0.5, 0.5, 0.9),
        ],
        "inputs":
        [PROJECT_ROOT / 'results' / _dataset / 'cate' / 'similarities.pkl'],
        "global_score":
        PROJECT_ROOT / 'results' / _dataset / 'bert' / similarity_name,
        "vocab":
        PROJECT_ROOT / 'data' / _dataset / 'vocab' / 'vocab.pkl',
        "n_words":
        10
    }
    # compute ensemble ranking
    logging.info(f"global similarities: {CONFIG['global_score']}")
    logging.info(f"local similarities: {CONFIG['inputs']}")
    ensemble_out_dirs = []
    for spec in CONFIG['specs']:
        local_w, global_w, rho = spec
        print(spec)
        # Sets the results folder
        ensemble_out_dir = PROJECT_ROOT / 'results' / _dataset / 'ensemble' / f'local-{local_w}-global-{global_w}-rho-{rho}'
        ensemble_out_dir.mkdir(parents=True, exist_ok=True)
        ensemble_out_dirs.append(ensemble_out_dir)
        # Run evaluation
        for input_fn in CONFIG['inputs']:
            run_ensemble_rankings(local_score=str(input_fn),
                                  global_score=str(CONFIG['global_score']),
                                  local_weight=local_w,
                                  global_weight=global_w,
                                  rho=rho,
                                  dry_run=False,
                                  out_dir=str(ensemble_out_dir))

    # Computing PMI NMPI and distinctiveness from this line below
    logging.warn(
        "Loading the co-occurence matrix can take serveral minutes depending on its size and may require 16G+ memory"
    )
    logging.warn(
        "If you don't have enough RAM, make sure you have enough swap sapce")
    cooccur_mat = pd.read_csv(PROJECT_ROOT / 'data' / _dataset /
                              'cooccurence.csv',
                              index_col=0)
    # Evaluation results output dir
    out_dir = PROJECT_ROOT / 'results' / _dataset / 'evaluations' / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    # redirects stdout to output file for later uees
    sys.stdout = open(out_dir / 'output.txt', 'w')
    vocab = load_vocab(CONFIG['vocab'], False)

    # Compute PMI, NPMI, Distinctivenesss
    for in_dir in ensemble_out_dirs:
        input_files = [str(f) for f in in_dir.glob('*.pkl')]
        out_file = out_dir / f"{in_dir.name}.json"
        evaluation(input_files,
                   cooccur_mat,
                   vocab,
                   out_file,
                   n_words=CONFIG['n_words'])
    sys.stdout.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--name',
        "-n",
        default="ensemble",
        help=
        "the name of the output folder, default to PROJECT_ROOT/results/DATASET/evalutaions/ensemble"
    )
    parser.add_argument("--dataset",
                        "-d",
                        required=True,
                        type=str,
                        help="the dataset folder")
    parser.add_argument(
        "--reduced_similarities",
        "-r",
        action="store_true",
        help=
        "will use similarities_reduced.pkl if this flag is specified otherwise will use similarities.pkl"
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, handlers=[RichHandler()])
    args = parse_args()
    main(args)
