from argparse import ArgumentParser
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from rich.logging import RichHandler
from utils.visualize import visualize_results

import logging


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--global_df',
                        '-g',
                        help="path to global consine similarities",
                        type=str,
                        default='./results/global_cos_similarities.csv')
    parser.add_argument('--local_df',
                        '-l',
                        help="path to local cosine similarities",
                        type=str,
                        required=True)
    parser.add_argument("--output",
                        "-o",
                        help='path to output',
                        type=str,
                        required=True)
    parser.add_argument("--dry_run",
                        "-d",
                        help='do not write output',
                        action='store_true')
    parser.add_argument("--local_weight",
                        help='weights of local embeddings',
                        default=0.5,
                        type=float)
    parser.add_argument("--global_weight",
                        help='weights of global embeddings',
                        default=0.5,
                        type=float)
    parser.add_argument("--rho",
                        help='weights of global embeddings',
                        default=0.5,
                        type=float)
    parser.add_argument('--verbose', "-v", action='store_true')
    return parser.parse_args()


def ensemble_ranking(score_g, score_l, rho, weight_global, weight_local):
    exponent = 1 / rho
    base = weight_global * np.power(
        1 / score_g, rho) + weight_local * np.power(1 / score_l, rho)
    return 1 / np.power(base, exponent)


def scale_data(data):
    scaler = MinMaxScaler()
    return scaler.fit_transform(data)


def read_csv(path):
    df = pd.read_csv(path, index_col=0)
    df.iloc[:, :] = MinMaxScaler().fit_transform(df)
    logging.info(f"loaded a csv with shape {df.shape} from {path}")
    return df


def rankings2df(rankings, vocab, topics):
    return pd.DataFrame(rankings, index=vocab, columns=topics)


def main(args):
    local_df = read_csv(args.local_df)
    global_df = read_csv(args.global_df)
    vocab = set(local_df.index) & set(global_df.index)
    dropped = set(local_df.index) - vocab
    if (len(dropped) > 0):
        logging.warning(f"dropped {list(dropped)} from local_df")
    dropped = set(global_df.index) - vocab
    if (len(dropped) > 0):
        logging.warning(f"dropped {list(dropped)} from global_df")
    local_df = local_df.loc[list(vocab)]
    global_df = global_df.loc[list(vocab)]
    rankings = ensemble_ranking(global_df,
                                local_df,
                                rho=args.rho,
                                weight_global=args.global_weight,
                                weight_local=args.local_weight)
    logging.info(
        f"Done ensemble ranking, min score {np.max(rankings)}, max score {np.min(rankings)})"
    )
    rankings_df = pd.DataFrame(rankings,
                               index=global_df.index,
                               columns=global_df.columns)
    visualize_results(rankings_df, 10)
    if not args.dry_run:
        logging.info(f"saving results to {args.output}")
        rankings_df.to_csv(args.output)


if __name__ == "__main__":
    args = parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.INFO, handlers=[RichHandler()])
    else:
        logging.basicConfig(level=logging.WARN, handlers=[RichHandler()])
    main(args)
