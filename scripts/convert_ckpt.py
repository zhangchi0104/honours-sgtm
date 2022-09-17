import torch
import argparse
from rich.console import Console
from rich.table import Table, Column
from collections import OrderedDict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", "-o", type=str, required=True)
    parser.add_argument("weights", type=str)
    return parser.parse_args()


def main(args):
    table = Table(
        Column("Old Key", style="cyan"),
        Column("New Key", style="magenta"),
    )
    console = Console()
    pl_state_dict = torch.load(args.weights, map_location='cuda')
    state_dict = pl_state_dict['state_dict']
    res = OrderedDict()
    for key in state_dict.keys():
        if key.startswith('model.bert.'):
            res[key[11:]] = state_dict[key]
            table.add_row(key, key[11:])
        elif key.startswith('model.'):
            res[key[6:]] = state_dict[key]
            table.add_row(key, key[6:])
        else:
            res[key] = state_dict[key]
            table.add_row(key, key)
    console.print(table)
    torch.save(res, args.output)


if __name__ == "__main__":
    args = parse_args()
    main(args)