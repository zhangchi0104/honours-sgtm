#%%
import pickle
import random

import torch
from transformers import BertTokenizer, BertForPreTraining, AdamW
from tqdm.auto import tqdm  # for our progress bar
BATCH_SIZE=8
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.eoncodings = encodings

    def __len__(self):
        return len(self.eoncodings['input_ids'])

    def __getitem__(self, i):
        return {key: torch.tensor(val[i]) for key, val in self.eoncodings.items() }
def main():
    inputs = None
    with open('./raw_inputs.pkl', 'rb') as f:
        inputs = pickle.load(f)
    dataset = MyDataset(inputs)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = BertForPreTraining.from_pretrained('bert-base-uncased')
    model.to(device)
    model.train()
    optim = AdamW(model.parameters(), lr=5e-5)

    epochs = 2

    for epoch in range(epochs):
        # setup loop with TQDM and dataloader
        loop = tqdm(loader, leave=True)
        for batch in loop:
            # initialize calculated gradients (from prev step)
            optim.zero_grad()
            # pull all tensor batches required for training
            input_ids = batch['input_ids'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            next_sentence_label = batch['next_sentence_label'].to(device)
            labels = batch['labels'].to(device)
            # process
            outputs = model(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            next_sentence_label=next_sentence_label,
                            labels=labels)
            # extract loss
            loss = outputs.loss
            # calculate loss for every parameter that needs grad update
            loss.backward()
            # update parameters
            optim.step()
            # print relevant info to progress bar
            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix(loss=loss.item())

    torch.save(model, 'bert-base-local.pth')

if __name__ == "__main__":
    main()
#%%
