from argparse import ArgumentParser
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import datetime


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--global_cos',
                        '-g',
                        help="path to global consine similarities",
                        type=str,
                        default='./results/global_cos_similarities.csv')
    parser.add_argument('--local_cos',
                        '-l',
                        help="path to local cosine similarities",
                        type=str,
                        required=True)
    parser.add_argument("--out_dir",
                        "-o",
                        help='path to output dir',
                        type=str,
                        default="./results/ensemble")
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
    return parser.parse_args()


def ensemble_ranking(score_g, score_l, rho, weight_global, weight_local):
    exponent = 1 / rho
    base = weight_global * np.power(
        1 / score_g, rho) + weight_local * np.power(1 / score_l, rho)
    return 1 / np.power(base, exponent)


def scale_data(data):
    scaler = MinMaxScaler()
    return scaler.fit_transform(data)


def compute_vocab_ensemble_rankings(score_g_df,
                                    score_l_df,
                                    vocab,
                                    rho=0.5,
                                    weight_global=0.5,
                                    weight_local=0.5):
    raw = np.zeros((len(vocab), score_l_df.shape[1])).astype(np.double)
    res = pd.DataFrame(raw, index=vocab, columns=score_g_df.columns)
    for topic_idx in range(score_l_df.shape[1]):
        score_l = score_l_df.loc[vocab, score_l_df.columns[topic_idx]]
        score_g = score_g_df.loc[vocab, score_g_df.columns[topic_idx]]
        res.iloc[:, topic_idx] = ensemble_ranking(score_g, score_l, rho,
                                                  weight_global, weight_local)
    return res


def main():
    args = parse_args()
    run_ensemble_rankings(local_score=args.local_cos,
                          global_score=args.global_cos,
                          local_weight=args.local_weight,
                          global_weight=args.global_weight,
                          rho=args.rho,
                          dry_run=args.dry_run,
                          out_dir=args.out_dir)


def run_ensemble_rankings(local_score, global_score, local_weight,
                          global_weight, rho, dry_run, out_dir):
    global_cos_df = pd.read_csv(global_score, index_col=0)
    local_cos_df = pd.read_csv(local_score, index_col=0)
    local_cos_df.loc[:, :] = scale_data(local_cos_df)
    global_cos_df.loc[:, :] = scale_data(global_cos_df)
    res = compute_vocab_ensemble_rankings(global_cos_df, local_cos_df,
                                          local_cos_df.index, rho,
                                          local_weight, global_weight)
    local_fn = local_score.split('/')[-1]
    local_fn = local_fn.split('.')[0]

    for topic in global_cos_df.columns:
        print(
            f"{topic}: {res[topic].sort_values(ascending=False).index[0:5].to_list()}"
        )
    if not dry_run:
        out_path = Path(out_dir) / f'ensemble_score_{local_fn}.csv'
        print(f"writing results to {out_path}")
        res.to_csv(out_path)


if __name__ == "__main__":
    main()
