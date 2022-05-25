# %%
import pickle
import datetime
from pathlib import Path
import torch
from transformers import BertTokenizer, BertForPreTraining, AdamW, BertConfig
from tqdm.auto import tqdm  # for our progress bar
import argparse
import wandb 
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, i):
        return {key: val[i] for key, val in self.encodings.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--out_dir', default=Path.cwd())
    parser.add_argument('-b', '--batch_size', type=int, default=8)
    parser.add_argument('-e', '--epochs', type=int, default=2)
    parser.add_argument('-t', '--tokens', type=str, default='tokens.pkl')
    parser.add_argument('-s', '--scratch', action='store_true')
    parser.add_argument('-n', '--name', default='')
    args = parser.parse_args()
    wandb.init(project="honours-sgtm")
    inputs = None
    with open(args.tokens, 'rb') as f:
        inputs = pickle.load(f)
    dataset = MyDataset(inputs)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    now_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    if args.scratch:
        model = BertForPreTraining(BertConfig())
        wandb.run.name = f"bert-local-scratch@{now_str}"
    else:
        model = BertForPreTraining.from_pretrained('bert-base-uncased')
        wandb.run.name = f"bert-local-scratch@{now_str}"
    wandb.config = {
        "epochs": args.epochs,
        "learning_rate": 5e-5,
        "batch_size": args.batch_size 
    }
    if args.name:
        wandb.run.name = args.name
    model.to(device)
    model.train()
    optim = AdamW(model.parameters(), lr=5e-5)

    for epoch in range(args.epochs):
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
            wandb.log({"loss": loss, "epoch": epoch})
            # calculate loss for every parameter that needs grad update
            loss.backward()
            # update parameters
            optim.step()
            # print relevant info to progress bar
            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix(loss=loss.item())

    model.save_pretrained(save_directory=args.out_dir)


if __name__ == "__main__": 
    main()
# %%
