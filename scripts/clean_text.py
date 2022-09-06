import argparse as a
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import WordNetLemmatizer
import re
from nltk.corpus import stopwords
from tqdm import tqdm
from typing import List
import spacy

spacy.prefer_gpu()


def parse_args():
    parser = a.ArgumentParser()
    parser.add_argument("file", type=str, help='path to plaintext file')
    parser.add_argument("--out", "-o", type=str)
    return parser.parse_args()


def clean_text(text: List[str], batch_size=200):
    res = []
    pipeline = spacy.load('en_core_web_trf')
    for lo in tqdm(range(0, len(text), batch_size)):
        hi = min(lo + batch_size, len(text))
        text_batch = text[lo:hi]
        _stopwords = set(stopwords.words('english'))
        processed_lines = list(pipeline.pipe(text_batch))
        for processed_line in processed_lines:
            tokens = [
                token.lemma_.lower() for token in processed_line
                if token.pos_ not in ['PUNCT', 'SPACE', 'SYM', "NUM"]
                and token.lemma_ not in _stopwords
            ]
            tokens = [
                token for token in tokens if re.match(r'^[a-zA-Z]+$', token)
            ]
            res.append(' '.join(tokens))
    return res


def drop_prefix(line: str):
    prefix_matcher = re.compile(r'^(afp|ap|)()')


def main():
    args = parse_args()

    f = open(args.file, 'r')
    lines = f.readlines()
    lines = [line.strip().lower().replace(r'\\', ' ') for line in lines if len(line) > 0]
    cleaned_text = clean_text(lines)
    f.close()

    f = open(args.out, 'w')
    f.write('\n'.join(cleaned_text))
    f.close()


if __name__ == "__main__":
    main()
