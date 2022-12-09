"""
Author: Chi Zhang
Licence: MIT

This script is for cleaning / preprocessing the text. 
Please refer README.md for detailed usages. 
"""

import argparse as a
import re
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
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
    """
    cleans the text by removing stopwords, punctuation, and lemmatizing the text
    Args:
        text (List[str]): list of documents, each document is a string
        batch_size (int): batch size for spacy pipeline

    Returns:
        List[str]: list of cleaned documents 
    """

    # Initialize the lemmatizer and pipeline
    res = []
    lemmatizer = WordNetLemmatizer()
    pipeline = spacy.load('en_core_web_trf')

    # Process the text in batches
    for lo in tqdm(range(0, len(text), batch_size)):
        hi = min(lo + batch_size, len(text))
        text_batch = text[lo:hi]
        _stopwords = set(stopwords.words('english'))

        processed_lines = list(
            pipeline.pipe(text_batch, disable=['ner', 'parser']))
        # Remove punctuations, spaces, numbers, stopwords and lemmatize the text
        for processed_line in processed_lines:
            tokens = [
                token for token in processed_line
                if token.pos_ not in ['PUNCT', 'SPACE', 'SYM', "NUM"]
                and token.lemma_ not in _stopwords
            ]
            # Lemmatize the text again with  nltk lemmatizer
            words = [token.lemma_.lower() for token in tokens]

            # Retag each word for nltk lemmatizer
            pos_dict = {
                "NOUN": "n",
                "ADJ": "a",
                "ADV": "r",
                "VERB": "v",
            }
            pos = [pos_dict.get(token.pos_, "n") for token in tokens]
            # remove words with length less than 3 and non-alphabetic characters
            words = [
                token for token in words
                if re.match(r'^[a-zA-Z_]+$', token) and len(token) >= 3
            ]
            # Lemmatize the text with WordNetLemmatizer
            tokens = [
                lemmatizer.lemmatize(word, p) for word, p in zip(words, pos)
            ]

            res.append(' '.join(tokens))
    return res


def main():
    args = parse_args()

    f = open(args.file, 'r', encoding="utf-8")
    lines = f.readlines()
    lines = [
        line.strip().lower().replace(r'\\', ' ') for line in lines
        if len(line) > 0
    ]
    cleaned_text = clean_text(lines)
    f.close()

    f = open(args.out, 'w')
    f.write('\n'.join(cleaned_text))
    f.close()


if __name__ == "__main__":
    main()
