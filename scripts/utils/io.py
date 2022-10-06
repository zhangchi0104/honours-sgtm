import pickle
import logging
from transformers import BertModel, BertTokenizer
import torch
import json


def load_vocab(path):
    vocab_set = None
    with open(path, 'rb') as f:
        vocab_set = pickle.load(f)

    vocab = [word.strip() for word in vocab_set]
    vocab = list(set(vocab))
    vocab = sorted(vocab)
    logging.info(f"Loaded {len(vocab)} words from {path}")
    return vocab


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


def load_model(path, device='cpu'):
    if path is None or path.strip() == '':
        logging.info("loading pretrained model")
        return BertModel.from_pretrained('bert-base-uncased')
    logging.info(f"loading BERT model from {path}")
    model = BertModel(con)
    weights = torch.load(path, map_location=device)
    model.to(device)
    missing_keys, unexpected_keys = model.load_state_dict(
        weights,
        strict=False,
    )
    logging.warn(f"the following keys are missing: {missing_keys}")
    logging.warn(f"the following keys are unexpected: {unexpected_keys}")
    return model.eval()


def load_tokenizer(path):
    if path is None or path.strip() == '':
        logging.info(f'loaded pretrained tokenizer')
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        return tokenizer
    logging.info(f"loading pretrained tokenizer from {path}")
    tokenizer = BertTokenizer(path)
    logging.info(
        f"The tokenizer has a vocabulary of size {len(tokenizer.vocab)}")
    return tokenizer


def load_dataset(path):
    f = open(path, "r")
    dataset = f.readlines()
    f.close()
    logging.info(f"loaded {len(dataset)} from {path}")
    return dataset
