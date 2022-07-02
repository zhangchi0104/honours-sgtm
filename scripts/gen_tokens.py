import pandas as pd
import random
import torch
import argparse
from tqdm import tqdm
from transformers import BertTokenizer
import pickle
from pathlib import Path


def generate_tokens(tokenizer, dataset, size=None):
    sentence_a = []
    sentence_b = []
    label = []
    num_sentences = 0
    loop = tqdm(dataset)
    loop.set_description("Generating tokens: ")
    for paragraph in loop:
        sentences = [
            sentence for sentence in paragraph.split('.') if sentence != ''
        ]
        num_sentences = len(sentences)
        if num_sentences > 1:
            start = random.randint(0, num_sentences - 2)
            # 50/50 whether is IsNextSentence or NotNextSentence
            if random.random() >= 0.5:
                # this is IsNextSentence
                sentence_a.append(sentences[start])
                sentence_b.append(sentences[start + 1])
                label.append(0)
            else:
                index = random.randint(0, len(dataset) - 1)
                # this is NotNextSentence
                sentence_a.append(sentences[start])
                sentence_b.append(dataset[index])
                label.append(1)
    print("Running tokenizers")
    inputs = tokenizer(sentence_a,
                       sentence_b,
                       return_tensors='pt',
                       max_length=512,
                       truncation=True,
                       padding='max_length')
    return inputs, label


def add_nsp_mlm(inputs, label):
    inputs['next_sentence_label'] = torch.LongTensor([label]).T
    inputs['labels'] = inputs.input_ids.detach().clone()
    rand = torch.rand(inputs.input_ids.shape)
    # create mask array
    mask_arr = (rand < 0.15) * (inputs.input_ids != 101) * \
               (inputs.input_ids != 102) * (inputs.input_ids != 0)

    selection = []
    loop = tqdm(range(inputs.input_ids.shape[0]))
    loop.set_description("Generating MLM: ")

    for i in loop:
        selection.append(torch.flatten(mask_arr[i].nonzero()).tolist())
    loop = tqdm(range(inputs.input_ids.shape[0]))
    loop.set_description("Setting labels: ")
    for i in loop:
        inputs.input_ids[i, selection[i]] = 3
    return inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab", type=str, default='')
    parser.add_argument("-o",
                        "--out_dir",
                        type=str,
                        default=Path.cwd() / 'data' / 'tokens')
    parser.add_argument('-s', "--size", type=int, default=None)
    parser.add_argument('-f',
                        '--file',
                        type=str,
                        default=Path.cwd() / 'data' / 'dataset.csv')
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    tokenizer = None
    if not args.vocab:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    else:
        tokenizer = BertTokenizer(vocab_file=args.vocab)
    data_f = open(args.file, 'r')
    dataset = data_f.readlines()
    data_f.close()
    inputs, labels = generate_tokens(tokenizer, dataset, args.size)
    name = f"tokens-{'pretrained' if not args.vocab else 'scratch'}-{tokenizer.vocab_size}.pkl"
    inputs = add_nsp_mlm(inputs, labels)
    
    with open(out_dir / name, 'wb') as f:
        print(f"Dumping tokens to file {args.out_dir}/{name}")
        pickle.dump(inputs, f)

    print("SUMMARY")
    print('=' * 80)
    print("Tokens:")
    for key, val in inputs.items():
        key = key + ":"
        print(f"{key:>20}{val.shape}")


if __name__ == '__main__':
    main()

# %%
