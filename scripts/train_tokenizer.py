import argparse
import glob

from tokenizers import BertWordPieceTokenizer

parser = argparse.ArgumentParser()
parser.add_argument(
    "--files",
    "-f",
    default=None,
    metavar="path",
    type=str,
    required=True,
    help="The files to use as training; accept '**/*.txt' type of patterns \
                          if enclosed in quotes",
)
parser.add_argument(
    "--out_dir",
    default="./",
    type=str,
    help="Path to the output directory, where the files will be saved",
)
parser.add_argument("--name",
                    default="bert-wordpiece",
                    type=str,
                    help="The name of the output vocab files")
parser.add_argument("--vocab_size",
                    default=10000,
                    type=int,
                    help="The name of the output vocab files")

args = parser.parse_args()

files = glob.glob(args.files)
if not files:
    print(f"File does not exist: {args.files}")
    exit(1)

# Initialize an empty tokenizer
tokenizer = BertWordPieceTokenizer(
    clean_text=True,
    handle_chinese_chars=True,
    strip_accents=True,
    lowercase=True,
)

# And then train
tokenizer.train(
    files,
    vocab_size=args.vocab_size,
    min_frequency=2,
    show_progress=True,
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
    limit_alphabet=1000,
    wordpieces_prefix="##",
)

# Save the files
tokenizer.save_model(args.out_dir, args.name)