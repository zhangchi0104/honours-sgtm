from lib2to3.pgen2.tokenize import tokenize
from transformers import BertTokenizer, BertModel


def dump_embeddings(tokenizer: BertTokenizer,
                    model: BertModel,
                    batch_size=10240):
    bert_vocab = tokenizer.get_vocab()
    all_words = list(bert_vocab.keys())
    for lo in range(0, len(all_words), batch_size):
        hi = min(len(all_words), batch_size + lo)
        tokens = 