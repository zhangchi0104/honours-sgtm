import argparse
import subprocess
from pathlib import Path
from ensemble_ranking import run_ensemble_rankings
from evaluation import evaluation
import pandas as pd
import datetime
import sys

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
    "inputs": [
        *(PROJECT_ROOT / 'results' / 'bert').glob("*"),
        (PROJECT_ROOT / 'results' / 'CatE' / '2022-07-03-09-36' /
         'cate-cos-similarities.csv')
    ],
    "baselines": [
        PROJECT_ROOT / 'results' / 'global_cos_similarities.csv',
        PROJECT_ROOT / 'results' / 'cate_cos_similarities.csv'
    ],
    "global_score":
    PROJECT_ROOT / 'results' / 'global_cos_similarities.csv',
    "n_words":
    10
}


def main():
    # compute ensemble ranking
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--name',
        default=f'run-{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")}')
    args = parser.parse_args()
    ensemble_out_dirs = []
    for spec in CONFIG['specs']:
        local_w, global_w, rho = spec
        print(spec)
        ensemble_out_dir = PROJECT_ROOT / 'results' / 'ensemble' / f'local-{local_w}-global-{global_w}-rho-{rho}'
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
    cooccur_mat = pd.read_csv(PROJECT_ROOT / 'data' / 'vocab' /
                              'cooccurence_matrix-agnews.csv',
                              index_col=0)
    now_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    out_dir = PROJECT_ROOT / 'results' / 'evaluations' / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    sys.stdout = open(out_dir / 'output.txt', 'w')

    for in_dir in ensemble_out_dirs:
        input_files = [str(f) for f in in_dir.glob('*')]
        out_file = out_dir / f"{in_dir.name}.json"
        evaluation(input_files,
                   cooccur_mat,
                   out_file,
                   n_words=CONFIG['n_words'])
    sys.stdout.close()


if __name__ == '__main__':
    main()
