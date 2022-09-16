import pickle
import logging
from transformers import BertModel, BertTokenizer
import torch


def load_vocab(path):
    vocab_set = None
    with open(path, 'rb') as f:
        vocab_set = pickle.load(f)

    vocab = [word.strip() for word in vocab_set]
    vocab = list(set(vocab))
    vocab = sorted(vocab)
    logging.info(f"Loaded {len(vocab)} words from {path}")
    return vocab


def load_seed(path):
    seeds = []
    with open(path, 'r') as f:
        seeds = f.readlines()
    seeds = [seed.strip() for seed in seeds]
    logging.info(f"loaded {len(seeds)} seeds from {path}")
    return seeds


def load_model(path):
    if path is None or path.strip() == '':
        logging.info("loading pretrained model")
        return BertModel.from_pretrained('bert-base-uncased')
    logging.info(f"loading BERT model from {path}")
    model = BertModel.from_pretrained('bert-base-uncased')
    weights = torch.load(path, map_location='cpu')
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
    return tokenizer