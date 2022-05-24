import pandas as pd
import random
import torch
import argparse
from transformers import BertTokenizer
import pickle
from pathlib import Path


def generate_tokens(tokenizer, dataset, size=None):
    news = dataset['description'][:size].tolist()
    inputs = tokenizer(news, return_tensors='pt',
                       max_length=512, truncation=True, padding='max_length')
    print(inputs.keys())
    labels = inputs.input_ids
    mask = inputs.attention_mask
    input_ids = labels.detach().clone()
    rand = torch.rand(input_ids.shape)
    mask_arr = (rand < .15) * (input_ids != 0) * (input_ids != 1) * (input_ids != 2)
    for i in range(input_ids.shape[0]):
    # get indices of mask positions from mask array
        selection = torch.flatten(mask_arr[i].nonzero()).tolist()
        # mask input_ids
        input_ids[i, selection] = 103
    return {
        'input_ids': input_ids,
        'attention_mask': mask,
        'labels': labels
    }



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", action='store_true')
    parser.add_argument("-o", "--out", type=str, default=Path.cwd() / 'tokens.pkl')
    parser.add_argument('-s', "--size", type=int, default=None)
    parser.add_argument('-d', '--data', type=str, default= Path.cwd() / 'data' / 'dataset.csv')
    args = parser.parse_args()
    tokenizer = None
    if args.pretrained:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    else:
        tokenizer = BertTokenizer(vocab_file='bert-vocab.txt')
    dataset = pd.read_csv('./data/dataset.csv')
    inputs = generate_tokens(tokenizer, dataset, args.size)

    with open(args.out, 'wb') as f:
        pickle.dump(inputs, f)

    print("SUMMARY")
    print('='*80)
    print("Tokens:")
    for key, val in inputs.items():
        print(f"{key:>20}\t:{val.shape}")


if __name__ == '__main__':
    main()

#%%
