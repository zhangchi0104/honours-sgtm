import os
import matplotlib.pyplot as plt
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    sub_parser = parser.add_subparsers(dest="command")
    ensemble_parser = sub_parser.add_parser("")