import numpy as np
import torch
from transformers import BertModel, BertTokenizer
from rich.progress import track


def get_bert_embeddings(vocab, model, tokenizer, batch_size=20, method='all'):
    embeddings = np.zeros((len(vocab), 768), dtype=np.float32)
    model = model.eval()

    for batch_idx_lo in track(range(0, len(vocab), batch_size),
                              description="Extracting embeddings from BERT"):
        batch_idx_hi = min(batch_idx_lo + batch_size, len(vocab))
        batch = vocab[batch_idx_lo:batch_idx_hi]
        inputs = tokenizer(batch, return_tensors='pt', padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            if method == 'all':
                outputs = outputs.last_hidden_state[:, 1:-1, :]
            else:
                outputs = outputs.last_hidden_state[:, 1, :]
            outputs = torch.squeeze(outputs, dim=1)
            outputs = outputs.numpy()
            embeddings[batch_idx_lo:batch_idx_hi, :] = outputs

    return embeddings


def batch_cosine_similarity(topic_emb, word_embs):
    return np.inner(topic_emb, word_embs) / (np.linalg.norm(topic_emb) *
                                             np.linalg.norm(word_embs, axis=1))


def cosine_similarity_with_topic(topic,
                                 vocab,
                                 tokenizer,
                                 model,
                                 batch_size=4,
                                 stem_weight=0.7,
                                 device='cpu'):
    topic_tokens: dict = tokenizer(topic, return_tensors='pt', padding=True)
    for k, v in topic_tokens.items():
        if isinstance(v, torch.Tensor):
            topic_tokens[k] = v.to(device)
    topic_emb = model(**topic_tokens)
    topic_emb = topic_emb.last_hidden_state[0, 1, :].detach().cpu().numpy()
    res = []
    res_embeddings = np.zeros((len(vocab), 768))
    for lo in track(range(0, len(vocab), batch_size),
                    description=f"Computing cosine similarites for {topic}"):
        hi = min(lo + batch_size, len(vocab))
        batch = vocab[lo:hi]
        inputs = tokenizer(batch, return_tensors='pt', padding=True)
        inputs.to(device)
        embeddings = None
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state.detach().cpu().numpy(
            )[:, 1:, :]
            res_embeddings = outputs.last_hidden_state.detach().cpu().numpy(
            )[:, 1, :]
        endings = torch.argwhere(inputs['input_ids'] == 102)
        endings[:, 1] = endings[:, 1] - 1
        for end in endings:
            row, col = end[0].item(), end[1].item()
            embedding = embeddings[row, :col, :]
            similarities = batch_cosine_similarity(topic_emb, embedding)
            if similarities.shape[0] == 1:
                res.append(similarities[0])
            else:
                weights_rest = (1 - stem_weight) / (similarities.shape[0] - 1)
                weights = [stem_weight]
                weights.extend([weights_rest] * (similarities.shape[0] - 1))
                res.append(np.average(similarities, weights=weights))
    print(res)
    return np.array(res), res_embeddings
