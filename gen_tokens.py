import pandas as pd
import random
import torch
import argparse
from transformers import BertTokenizer
import pickle
def generate_tokens(tokenizer, dataset):
    sentence_a = []
    sentence_b = []
    label = []
    num_sentences = 0
    news = dataset['description']
    for  paragraph in news.iteritems():
        sentences = [
            sentence for sentence in paragraph.split('.') if sentence != ''
        ]
        num_sentences = len(sentences)
    if num_sentences > 1:
        start = random.randint(0, num_sentences-2)
        # 50/50 whether is IsNextSentence or NotNextSentence
        if random.random() >= 0.5:
            # this is IsNextSentence
            sentence_a.append(sentences[start])
            sentence_b.append(sentences[start+1])
            label.append(0)
        else:
            index = random.randint(0, dataset.shape[0] - 1)
            # this is NotNextSentence
            sentence_a.append(sentences[start])
            sentence_b.append(dataset['description'][index])
            label.append(1)
        inputs = tokenizer(sentence_a, sentence_b, return_tensors='pt',
                   max_length=512, truncation=True, padding='max_length')
        return inputs, label

def add_nsp_mlm(inputs, label):
    inputs['next_sentence_label'] = torch.LongTensor([label]).T
    inputs['labels'] = inputs.input_ids.detach().clone()
    rand = torch.rand(inputs.input_ids.shape)
    # create mask array
    mask_arr = (rand < 0.15) * (inputs.input_ids != 101) * \
           (inputs.input_ids != 102) * (inputs.input_ids != 0)

    selection = []

    for i in range(inputs.input_ids.shape[0]):
        selection.append(
            torch.flatten(mask_arr[i].nonzero()).tolist()
        )
    for i in range(inputs.input_ids.shape[0]):
        inputs.input_ids[i, selection[i]] = 103
    return inputs

def main(): 
    parser = argparse.ArgumentParser() 
    parser.add_argument("--pretrained", action='store_true')
    args = parser.parse_args()
    tokenizer = None
    if (args.pretrained):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    else:
        tokenizer = BertTokenizer(vocab_file='bert_vocab.txt')
    dataset = pd.read_csv('./data/dataset.csv')
    inputs, labels = generate_tokens(tokenizer, dataset)
    inputs = add_nsp_mlm(inputs,labels)
    with open('tokens.pkl' , 'wb') as f:
        pickle.dump(inputs, f)

if __name__ == '__main__':
    main()

    
