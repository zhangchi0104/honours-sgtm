from argparse import ArgumentParser
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from transformers import BertModel, BertTokenizer
from tqdm.auto import tqdm
import re


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--file",
                        type=str,
                        help="The path to the input file",
                        required=True)
    parser.add_argument("--type",
                        "-t",
                        type=str,
                        help="the type of input file",
                        required=True)
    parser.add_argument("--out_dir",
                        "-o",
                        type=str,
                        help="The path to the output directory")
    parser.add_argument("--vocab",
                        "-v",
                        type=str,
                        help="The path to the vocab file")
    parser.add_argument("--topics", "-t", action='append', help="The topics")
    parser.add_argument("--local_model",
                        action='store_true',
                        help="Idenfity as local model")
    return parser.parse_args()


def cos_similarities_batch(topic, words):
    return np.inner(
        topic, words) / (np.linalg.norm(topic) * np.linalg.norm(words, axis=1))


def comppute_bert_cosine_similarities(vocab,
                                      topic,
                                      tokenizer,
                                      model,
                                      batch_size=4):

    def get_embeddings_batch(model, tokens):
        with torch.no_grad():
            output = model(**tokens)
            embedding = output.last_hidden_state[:, 1, :]
            return embedding

    def get_embeddings(model, tokens, embedding_size=768):
        with torch.no_grad():
            output = model(**tokens)
            embedding = output.last_hidden_state[0][1]
        return torch.reshape(embedding, (embedding_size, ))

    res_col = np.zeros((len(vocab), ))
    vocab = list(vocab)
    loop = tqdm(range(0, len(vocab), batch_size))
    loop.set_description(f"topic: {topic}")
    topic_token = tokenizer(topic,
                            return_tensors='pt',
                            padding=True,
                            max_length=10,
                            truncation=True)
    topic_emb = get_embeddings(model, topic_token)
    for batch_index in loop:
        lo = batch_index
        hi = min(batch_index + batch_size, len(vocab))
        batch = vocab[batch_index:batch_index + batch_size]
        tokens = tokenizer(batch,
                           return_tensors='pt',
                           padding='max_length',
                           max_length=10,
                           truncation=True)
        # if len(token['input_ids']) > 3:
        #     print(f"WARNING: Word '{word}' is not in BERT's vocabulary")
        word_embs = get_embeddings_batch(model, tokens)
        res_col[lo:hi] = cos_similarities_batch(topic_emb, word_embs)
        # res_col.append(cosine(topic_emb, word_emb))
    return res_col


def generate_dataframe(similarities, vocab, topics):
    arr = np.array(similarities)
    arr = arr.T
    res_dict = {
        '_vocab': list(vocab),
    }
    for i, topic in enumerate(topics):
        res_dict[topic] = arr[:, i]
    res_df = pd.DataFrame(res_dict)
    res_df = res_df.set_index(['_vocab'])
    return res_df


def prepare_bert_inputs(in_dir: str):
    model_name = in_dir
    model_type, vocab_size = re.findall(r"bert-(pretrained|scratch)-(\d+)",
                                        model_name)[0]
    if model_type == 'pretrained':
        model = BertModel.from_pretrained('bert-base-uncased')
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    else:
        vocab_filename = f"./data/vocab/bert-local-{vocab_size}.txt"
        tokenizer = BertTokenizer(vocab_file=vocab_filename)
        model = BertModel.from_pretrained(model_name)

    return model, tokenizer, (model_type, vocab_size)


def main():
    args = parse_args()
    with open(args.vocab, 'rb') as f:
        vocab = pickle.load(f)
    if args.type.lower() == 'bert':
        model, tokenizer, metadata = prepare_bert_inputs(args.file)
        model_type, vocab_size = metadata
        res = []
        for topic in args.topics:
            res_col = comppute_bert_cosine_similarities(vocab,
                                                        topic,
                                                        tokenizer,
                                                        model,
                                                        batch_size=256)
            res.append(res_col)
        res_df = generate_dataframe(res, vocab, args.topics)
        res_df.to_csv(
            Path(args.out_dir) /
            f"{args.type}-local-{model_type}-{vocab_size}.csv")
    elif args.type.lower() == 'cate':
        return
