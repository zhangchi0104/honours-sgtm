import pandas as pd
import numpy as np
import logging
from rich.logging import RichHandler
import matplotlib.pylab as plt
from os.path import join as join_path
from scripts.utils.io import load_seed
import re
from pathlib import Path
import os

import seaborn as sns

FN_MATCHER = r'local-(?P<local>[0-9]\.[0-9])-global-(?P<global>[0-9]\.[0-9])-rho-(?P<rho>[0-9]\.[0-9]).json'
LINE_MATCHER = r'(?P<seed>[a-z_]+):(?P<pmi>[0-9.]*)'
import json

SAVE_DIR = './plots'
os.makedirs(SAVE_DIR, exist_ok=True)
logger = logging.getLogger('plots.py')


def aggregated_plots_by_metrics(data: dict, name: str, fig_path):
    """
    Plots the line plots of performance
        - Each chart contains a performance metric for a dataset
    """
    plt.figure()

    xs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for k, ys in data.items():
        plt.xlabel('Local Knowledge Weight')
        plt.title(name)
        plt.plot(xs, ys, label=k)
    plt.legend()
    logger.info(f"Saving plot \"{name}\" to {fig_path}")
    plt.savefig(fig_path)


def result_json2dataframe(datasets):
    from os.path import join as j
    metrics = ['pmi', 'npmi', 'distinctiveness']
    results = {
        "dataset": [],
        "method": [],
        "metric": [],
        "score": [],
    }

    for dataset in datasets:
        evaluation_root = f"./results/{dataset}/evaluations/"
        methods = [
            ('CaTE', j(evaluation_root, 'cate.json')),
            ('Finetuned BERT', j(evaluation_root, 'finetuned-bert.json')),
            ('Pretrained BERT', j(evaluation_root, 'pretrained-bert.json')),
            ('SeededLDA', j(evaluation_root, 'seeded_lda.json')),
            (
                'EnsembleTM',
                j(evaluation_root, 'ensemble',
                  'local-0.5-global-0.5-rho-0.5.json'),
            ),
            (
                'EnsembleTM (Rank Reduced)',
                j(evaluation_root, 'evaluation_reduced',
                  'local-0.5-global-0.5-rho-0.5.json'),
            ),
        ]
        for method, file in methods:
            try:
                f = open(file, 'r')
                result = json.load(f)
                f.close()
                if len(result.keys()) == 1:
                    key = list(result.keys())[0]
                    result = result[key]

                for metric in metrics:
                    results['dataset'].append(dataset)
                    results['method'].append(method)
                    results['metric'].append(metric)
                    results['score'].append(result[metric])
            except FileNotFoundError:
                for metric in metrics:
                    results['dataset'].append(dataset)
                    results['method'].append(method)
                    results['metric'].append(metric)
                    results['score'].append(None)
    df = pd.DataFrame(results)
    return df


def parse_ensemble_results(dir):
    files = list(dir.glob('local-*.json'))
    res = np.zeros((len(files), 6))
    for i, filename in enumerate(files):
        matches = re.search(FN_MATCHER, str(filename))
        f = open(filename, 'r')
        data = json.load(f)
        res[i][0] = float(matches.group('global'))
        res[i][1] = float(matches.group('local'))
        res[i][2] = float(matches.group('rho'))
        res[i][3] = float(data['scores']['pmi'])
        res[i][4] = float(data['scores']['npmi'])
        res[i][5] = float(data['scores']['distinctiveness'])
    df = pd.DataFrame(
        res,
        columns=[
            'global_weight', 'local_weight', 'rho', "pmi", 'npmi',
            'distinctiveness'
        ],
    )
    return df


def ensemble_pmi2dataframe(*datasets):
    # load seeds
    data = {"Dataset": [], "Type": [], "PMI": []}
    for dataset in datasets:

        dataset_result = {
            "In vocabulary Average": 0,
            "Out Vocabulary Average": 0,
            "Overall Average": 0,
        }
        seed_path = os.path.join('./data', dataset, 'seeds.json')
        result_path = os.path.join('./results/', dataset, 'evaluations',
                                   'ensemble_pmi.txt')
        in_vocab, out_vocab = load_seed(seed_path, False)
        f = open(result_path, 'r')
        lines = f.readlines()
        f.close()
        pmis = ensemble_pmi2dict(lines)
        out_vocab_pmi = sum(pmis[word] for word in out_vocab) / len(in_vocab)
        in_vocab_pmi = sum(pmis[word] for word in in_vocab) / len(out_vocab)
        dataset_result['In vocabulary Average'] = in_vocab_pmi
        dataset_result['Out Vocabulary Average'] = out_vocab_pmi
        dataset_result['Overall Average'] = (
            sum(pmis[word]
                for word in out_vocab) + sum(pmis[word]
                                             for word in in_vocab)) / len(pmis)
        for key, value in dataset_result.items():
            data['Dataset'].append(dataset)
            data['Type'].append(key)
            data['PMI'].append(value)
    print(data)
    return pd.DataFrame(data)


def ensemble_pmi2dict(lines):
    res = {}
    for line in lines:
        match = re.match(LINE_MATCHER, line)
        match = match.groupdict()
        seed = match['seed']
        pmi = match['pmi']
        res[seed] = float(pmi)

    return res


def main():
    DATASETS = ['scidoc', 'yelp', '20NewsGroup', "dbpedia"]
    # PMI without rank reduction
    PLOT_NAME = [
        "{} vs Local Knowledge weight WITH rank reduction",
        "{} vs Local Knowledge weight WITHOUT rank reduction",
    ]
    FIG_SUFFIX = ['normal', 'reduced']

    for i, results_root in enumerate(["ensemble", "evaluation_reduced"]):
        results = {}
        # collect results to DF
        for dataset in DATASETS:
            result_dir = Path(
                './results') / dataset / 'evaluations' / results_root
            results[dataset] = parse_ensemble_results(result_dir)
        pmis = {}
        npmis = {}
        distinctiveness = {}
        for key, df in results.items():
            pmis[key] = df[df.rho == 0.5].sort_values('local_weight')['pmi']
            npmis[key] = df[df.rho == 0.5].sort_values('local_weight')['npmi']
            distinctiveness[key] = df[df.rho == 0.5].sort_values(
                'local_weight')['distinctiveness']
        suffix = FIG_SUFFIX[i]
        aggregated_plots_by_metrics(
            pmis,
            PLOT_NAME[i].format("PMI"),
            f'./plots/pmi_{suffix}.png',
        )
        aggregated_plots_by_metrics(
            npmis,
            PLOT_NAME[i].format("NPMI"),
            f'./plots/npmi_{suffix}.png',
        )
        aggregated_plots_by_metrics(
            distinctiveness,
            PLOT_NAME[i].format("Distinctiveness"),
            f'./plots/distinctiveness_{suffix}.png',
        )

    # [Bar charts] Comparsions of methods By datasets
    results = result_json2dataframe(DATASETS)
    metrics = ['pmi', 'distinctiveness']
    for metric in metrics:
        plot = sns.catplot(results[results['metric'] == metric],
                           kind='bar',
                           x='dataset',
                           y='score',
                           hue='method')

        plot = plot.fig
        path = f'./plots/{metric}_by_methods.png'
        logging.info(f"Saving bar plots of {metric} to {path}")
        plot.savefig(path)

    # Bar charts for invcab and out PMI for ensemble methods
    results = ensemble_pmi2dataframe(DATASETS)
    plot = sns.catplot(results, kind='bar', x='Dataset', y='PMI', hue='Type')
    plot = plot.fig
    logging.info("Saving in vocab vs out vocab figure")
    plot.savefig("./plots/in_vocab_vs_out_vocab.png")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, handlers=[RichHandler()])
    main()