import argparse
import subprocess
from pathlib import Path
from ensemble_ranking import run_ensemble_rankings
from evaluation import evaluation
import pandas as pd
import datetime
import sys
import logging
from rich.logging import RichHandler
from utils.io import load_vocab


def main(args):
    DATASET = args.dataset
    similarity_name = 'similarities.pkl' if not args.reduced_similarities else 'similarities_reduced.pkl'
    PROJECT_ROOT = Path(
        "~/code/github.com/zhangchi0104/honours-sgtm").expanduser().absolute()
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
        [PROJECT_ROOT / 'results' / DATASET / 'cate' / 'similarities.pkl'],
        "global_score":
        PROJECT_ROOT / 'results' / DATASET / 'bert' / similarity_name,
        "vocab":
        PROJECT_ROOT / 'data' / DATASET / 'vocab' / 'vocab.pkl',
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
        ensemble_out_dir = PROJECT_ROOT / 'results' / DATASET / 'ensemble' / f'local-{local_w}-global-{global_w}-rho-{rho}'
        ensemble_out_dir.mkdir(parents=True, exist_ok=True)
        ensemble_out_dirs.append(ensemble_out_dir)
        for input_fn in CONFIG['inputs']:
            run_ensemble_rankings(local_score=str(input_fn),
                                  global_score=str(CONFIG['global_score']),
                                  local_weight=local_w,
                                  global_weight=global_w,
                                  rho=rho,
                                  dry_run=False,
                                  out_dir=str(ensemble_out_dir))
    cooccur_mat = pd.read_csv(PROJECT_ROOT / 'data' / DATASET /
                              'cooccurence.csv',
                              index_col=0)

    out_dir = PROJECT_ROOT / 'results' / DATASET / 'evaluations' / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    sys.stdout = open(out_dir / 'output.txt', 'w')
    vocab = load_vocab(CONFIG['vocab'], False)
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
        default=f'run-{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")}',
    )
    parser.add_argument("--dataset", "-d", required=True, type=str)
    parser.add_argument("--reduced_similarities", "-r", action="store_true")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, handlers=[RichHandler()])
    args = parse_args()
    main(args)
