import torch
from tqdm.auto import tqdm
from transformers import BertTokenizer, BertConfig, BertModel, AdamW
import pandas as pd
device = device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

dataset = pd.read_csv('./data/dataset.csv')
my_tokenizer = BertTokenizer(vocab_file='./bert-vocab.txt')
tokens = my_tokenizer(dataset['description'].values.tolist(),
                      return_tensors='pt',
                      max_length=512,
                      padding='max_length',
                      truncation=True,
                      )
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.eoncodings = encodings

    def __len__(self):
        return len(self.eoncodings['input_ids'])

    def __getitem__(self, i):
        return {key: val[i] for key, val in self.eoncodings.items() }

dataloader = torch.utils.data.DataLoader(MyDataset(tokens), batch_size=64, shuffle=True)
bert_config = BertConfig(
    vocab_size=10000,
    max_position_embeddings=512,
    hidden_size=512,
    num_attention_heads=12,
    num_hidden_layers=5,
    type_vocab_size=1,
)
model = BertModel(config=bert_config)
model.to(device)
optim = AdamW(model.parameters(), lr=1e-4)


epochs = 2

for epoch in range(epochs):
    # setup loop with TQDM and dataloader
    loop = tqdm(dataloader, leave=True)
    torch.cuda.empty_cache()
    for batch in loop:
        # initialize calculated gradients (from prev step)
        optim.zero_grad()
        # pull all tensor batches required for training
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['token_type_ids'].to(device)
        # process
        outputs = model(input_ids, attention_mask=attention_mask,
                        token_type_ids=labels)
        # extract loss
        loss = outputs.loss
        # calculate loss for every parameter that needs grad update
        loss.backward()
        # update parameters
        optim.step()
        # print relevant info to progress bar
        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(loss=loss.item())