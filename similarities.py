import argparse
import os, sys
import subprocess
from unicodedata import name


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--method', '-m', choices=['bert', 'cate'])
    parser.add_argument('--dataset', '-d', required=True)
    parser.add_argument('--use_embeddings', "-e", action='store_true')
    parser.add_argument(
        '--output',
        nargs='?',
        choices=['words', 'embeddings'],
        default=[],
    )
    parser.add_argument("--reduced_embeddings", "-r", action='store_true')
    parser.add_argument('--dry_run', action='store_true')
    return parser.parse_args()


def main(args):
    dataset = args.dataset
    method = args.method
    proc_args = ['python3', 'scripts/similarities.py']
    datapath = os.path.join('data', dataset)
    results_path = os.path.join('results', dataset, method)
    vocab_path = os.path.join(datapath, 'vocab', 'vocab.pkl')
    seeds_path = os.path.join(datapath, 'seeds.json')
    out_similarities_name = 'similarities.pkl' if not args.reduced_embeddings else 'similarities_reduced.pkl'
    similarities_csv_path = os.path.join(results_path, out_similarities_name)
    proc_args.extend([
        f"--output={similarities_csv_path}",
        f"--vocab={vocab_path}",
        f"--seeds={seeds_path}",
    ])
    if 'words' in args.output:
        words_path = os.path.join(results_path, "word_sets.json")
        proc_args.append(f"--out_words={words_path}")
    if "embeddings" in args.output:
        out_embeddings_path = os.path.join(results_path, "embeddings.pkl")
        proc_args.append(f"--out_embeddings={out_embeddings_path}")
    if args.use_embeddings:
        embedding_name = "embeddings.pkl" if args.reduced_embeddings else "embeddings_reduced.pkl"
        input_emb_path = os.path.join(results_path, embedding_name)
        proc_args.append(f'--embeddings={input_emb_path}')
    proc_args.append(args.method)
    if args.method == 'cate':
        topics_path = os.path.join(results_path, 'emb_seeds_t.txt')
        emb_path = os.path.join(results_path, 'emb_seeds_w.txt')
        proc_args.extend([f'--topic={topics_path}'])
        proc_args.extend([f"--words={emb_path}"])
    elif args.method == 'bert':
        base_model_path = os.path.join('models', dataset, 'bert')
        weights_path = os.path.join(base_model_path, 'finetuned-model')
        tokenizer_path = os.path.join(base_model_path, 'tokenizer')

        proc_args.extend([
            f'--weights={weights_path}',
            f'--tokenizer={tokenizer_path}',
        ])

    if args.dry_run:
        args_str = ' '.join(proc_args[:2]) + ' \\\n' + ' \\\n'.join(
            proc_args[2:])
        print(args_str)
    else:
        subprocess.run(args=proc_args, stdout=sys.stdout, stderr=sys.stderr)


if __name__ == "__main__":
    args = parse_args()
    main(args)