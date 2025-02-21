import pickle
import logging
from transformers import BertModel, BertTokenizer
import torch
import json


def load_vocab(path, keys_only=True):
    vocab_dict = None
    with open(path, 'rb') as f:
        vocab_dict = pickle.load(f)
        logging.info(f"Loaded {len(vocab_dict)} words from {path}")
    if keys_only:
        vocab = [word.strip() for word in vocab_dict]
        vocab = list(set(vocab))
        vocab = sorted(vocab)
        return vocab
    else:
        return vocab_dict


def load_seed(path, combine_result=True):
    f = open(path, "r")
    topics = json.load(f)
    in_vocab, out_vocab = topics['in_vocab'], topics['out_vocab']
    logging.info(
        f"Loaded {len(in_vocab)} in-vocab words and {len(out_vocab)} out-vocab words from {path}"
    )
    if combine_result:
        return [*in_vocab, *out_vocab]
    return in_vocab, out_vocab


def load_model(path=None, device='cpu'):
    if path is None:
        model = BertModel.from_pretrained('bert-base-uncased')
    else:
        model = BertModel.from_pretrained(path)
    model.eval()
    model.to(device)
    return model


def load_tokenizer(path=None):
    if path is None:
        logging.info("Loading pretrained tokenizer")
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    else:
        logging.info(f"loading tokenizer from {path}")
        tokenizer = BertTokenizer.from_pretrained(path)
    return tokenizer


def load_dataset(path):
    f = open(path, "r")
    dataset = f.readlines()
    f.close()
    logging.info(f"loaded {len(dataset)} from {path}")
    return dataset
