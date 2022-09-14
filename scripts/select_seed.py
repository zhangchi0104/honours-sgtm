import argparse
import re
from sklearn.cluster import KMeans
import numpy as np
import logging
from gensim.models import Word2Vec, KeyedVectors
from transformers import BertTokenizer, BertModel
import random
from rich.console import Console
from rich.logging import RichHandler
from utils.embeddings import get_bert_embeddings

logging.basicConfig(level=logging.INFO, handlers=[RichHandler()])
logger = logging.getLogger("select_seed")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--embeddings", type=str)
    parser.add_argument("--output", type=str, default="seed.txt")
    parser.add_argument("--n_in_vocab", type=int, default=10)
    parser.add_argument("--n_out_vocab", type=int, default=10)
    parser.add_argument("--tokenizer_vocab", type=str, default="")
    parser.add_argument("--output_vocab_embedings",
                        type=str,
                        default="vocab_embeddings.bin")

    return parser.parse_args()


def main(args):
    documents = None
    console = Console()
    with open(args.dataset, "r") as f:
        documents = f.readlines()
    documents = [doc.strip().split(' ') for doc in documents]
    logger.info("Loaded {} documents".format(len(documents)))
    # get document vocabulary
    if args.embeddings:
        logger.info("Loading embeddings from {}".format(args.embedding))
        doc_vocab_embeddings = load_word2vec_embeddings(args.embedding)
    else:
        logger.info("\"--embeddings\" not set, Training embeddings")
        model = train_embeddings(documents)
        model.wv.save(args.output_vocab_embedings)
        doc_vocab_embeddings = model.wv

    # get tokenizer vocabulary
    if args.tokenizer_vocab:
        tokenizer_vocab = load_tokenizer_vocab(args.tokenizer_vocab)
    else:
        tokenizer_vocab = extract_bert_vocab()

    doc_vocab_set = set(doc_vocab_embeddings.index_to_key)
    tokenizer_vocab_set = set(tokenizer_vocab)
    doc_vocab_set = tokenizer_vocab_set & doc_vocab_set
    tokenizer_vocab_set = tokenizer_vocab_set - doc_vocab_set  # in tokenizer but not in doc
    # in both
    tokenizer_vocab = list(tokenizer_vocab_set)
    doc_vocab = list(doc_vocab_set)
    # extract embeddings for tokenizer vocab
    model = BertModel.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer_embeddings = get_bert_embeddings(tokenizer_vocab, model,
                                               tokenizer)
    doc_vocab_embeddings = get_doc_embeddings(doc_vocab_embeddings, doc_vocab)
    # cluster tokenizer vocab
    doc_vocab_labels = kmeans_cluster(doc_vocab_embeddings, args.n_in_vocab)
    tokenizer_labels = kmeans_cluster(tokenizer_embeddings, args.n_out_vocab)
    # saving
    doc_seeds = select_seeds(doc_vocab, doc_vocab_labels)
    tokenizer_seeds = select_seeds(tokenizer_vocab, tokenizer_labels)
    console.print("Document seeds: {}".format(doc_seeds))
    console.print("Tokenizer seeds: {}".format(tokenizer_seeds))
    logger.info("Saving seeds to {}".format(args.output))
    with open(args.output, "w") as f:
        f.write("\n".join(doc_seeds))
        f.write("\n")
        f.write("\n".join(tokenizer_seeds))
    return


def select_seeds(vocab, labels):
    clusters = np.unique(labels)
    res = []
    for cluster in clusters:
        word_idx = np.where(labels == cluster)[0].tolist()
        idx = random.choice(word_idx)
        res.append(vocab[idx])
    return res


def train_embeddings(docs):
    logger.info("Training embeddings")
    model = Word2Vec(docs, vector_size=100, window=5, min_count=5, workers=4)
    # print(model.wv.index_to_key)
    return model


def get_doc_embeddings(word_vec, vocab):
    embeddings = np.zeros((len(vocab), 100), dtype=np.float32)
    for i, word in enumerate(vocab):
        embeddings[i, :] = word_vec[word]
    return embeddings


def load_tokenizer_vocab(path):
    lines = None
    with open(path, 'r') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    words = [line for line in lines if re.match(r'^[a-zA-Z]+$', line)]
    logger.info("Loaded {} words from {}".format(len(words), path))
    return words


def kmeans_cluster(embeddings, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters).fit(embeddings)
    return kmeans.labels_


def extract_bert_vocab():
    tokenizer: BertTokenizer = BertTokenizer.from_pretrained(
        'bert-base-uncased')
    vocab = tokenizer.vocab
    vocab = list(vocab.keys())
    vocab = [word for word in vocab if re.match(r'^[a-zA-Z]+$', word)]
    logger.info("laoded {} words from bert-base-uncased tokenizer".format(
        len(vocab)))
    return vocab


def load_word2vec_embeddings(path):
    wv = KeyedVectors.load(path)
    return wv


if __name__ == "__main__":
    args = parse_args()
    main(args)