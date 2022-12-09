from argparse import ArgumentParser
from unittest import result
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from rich.logging import RichHandler
from utils.visualize import visualize_results
import os
import logging


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--global_df',
                        '-g',
                        help="path to global consine similarities",
                        type=str,
                        default='./results/global_cos_similarities.csv',
                        required=True)
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
    args = parser.parse_args()
    return args

def ensemble_ranking(score_g, score_l, rho, weight_global, weight_local):
    """
    Combines the scores of local and global embeddings with the formula

    Args:
        score_g (np.array): global scores
        score_l (np.array): local scores
        rho (float): exponent
        weight_global (float): weight of global scores
        weight_local (float): weight of local scores
    Returns:
        np.array: combined scores
    """
    exponent = 1 / rho
    base = weight_global * np.power(
        1 / score_g, rho) + weight_local * np.power(1 / score_l, rho)
    return 1 / np.power(base, exponent)


def scale_data(data):
    """
    Remove the negative values with minmax scaler

    Args:
        data (np.array): data to be scaled to [0, 1]
    Returns:
        np.array: scaled data
    """
    scaler = MinMaxScaler()
    return scaler.fit_transform(data)


def read_csv(path):
    """
    Read a csv file and scale the data

    Args: 
        path (str): path to csv file
    Returns:
        pd.DataFrame: scaled data
    """
    df = pd.read_pickle(path)
    df.iloc[:, :] = MinMaxScaler().fit_transform(df)
    logging.info(f"loaded a csv with shape {df.shape} from {path}")
    return df


def rankings2df(rankings, vocab, topics):
    """
    Convert rankings to a dataframe
    
    Args:
        rankings (np.array): rankings
        vocab (list): vocabulary
        topics (list): topics
    Returns:
        pd.DataFrame: rankings
    """
    return pd.DataFrame(rankings, index=vocab, columns=topics)


def run_ensemble_rankings(
    global_score: str,
    local_score: str,
    global_weight: float,
    local_weight: float,
    rho: float,
    out_dir: str,
    dry_run=False,
):
    """
    Run ensemble ranking 
    Args:
        global_score (str): path to global scores
        local_score (str): path to local scores
        global_weight (float): weight of global scores
        local_weight (float): weight of local scores
        rho (float): exponent
        out_dir (str): path to output directory
        dry_run (bool): flat to indicate that should write output
    Returns:
        None
    """
    local_df = read_csv(local_score)
    global_df = read_csv(global_score)
    vocab = set(local_df.index) & set(global_df.index)
    # dropped = set(local_df.index) - vocab
    # if (len(dropped) > 0):
    #     logging.warning(f"dropped {list(dropped)} from local_df")
    # dropped = set(global_df.index) - vocab
    # if (len(dropped) > 0):
    #    logging.warning(f"dropped {list(dropped)} from global_df")
    local_df = local_df.loc[list(vocab)]
    global_df = global_df.loc[list(vocab)]
    rankings = ensemble_ranking(global_df,
                                local_df,
                                rho=rho,
                                weight_global=global_weight,
                                weight_local=local_weight)
    duplicate_masks = rankings.index.duplicated(keep=False)
    duplicate_idx = np.argwhere(duplicate_masks == True)
    duplicate_idx = duplicate_idx.flatten().tolist()
    dup_words = list(rankings.index[duplicate_idx])
    words = ' '.join(
        [f"{word}@{idx}" for idx, word in zip(duplicate_idx, dup_words)])
    logging.warning(f"{words} are duplicated, keeping first occurences")
    rankings = rankings[~rankings.index.duplicated(keep='first')]
    # logging.info(
    #     f"Done ensemble ranking, min score {np.max(rankings)}, max score {np.min(rankings)})"
    # )
    rankings_df = pd.DataFrame(rankings,
                               index=global_df.index,
                               columns=global_df.columns)
    visualize_results(rankings_df, 10)

    if not dry_run:
        results_path = os.path.join(out_dir, 'scores.pkl')
        logging.info(f"saving results to {results_path}")
        rankings_df.to_pickle(results_path)


def main(args):
    logging.info(args)
    run_ensemble_rankings(
        global_score=args.global_df,
        local_score=args.local_df,
        global_weight=args.global_weight,
        local_weight=args.local_weight,
        rho=args.rho,
        dry_run=args.dry_run,
        out_dir=args.output,
    )


if __name__ == "__main__":
    args = parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.INFO, handlers=[RichHandler()])
    else:
        logging.basicConfig(level=logging.WARN, handlers=[RichHandler()])
    main(args)
