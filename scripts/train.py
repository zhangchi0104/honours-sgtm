from pathlib import Path
import pickle
import re
import torch.cuda
from torch.utils.data import Dataset
from transformers import BertForPreTraining, Trainer, TrainingArguments, BertConfig
from transformers import BertTokenizer
import wandb
import datetime, argparse

class AgNewsDataset(Dataset):
    def __init__(self, encodings):
        self.eoncodings = encodings

    def __len__(self):
        return len(self.eoncodings['input_ids'])

    def __getitem__(self, i):
        return {key: torch.tensor(val[i]) for key, val in self.eoncodings.items()}

def train(out, dataset, model, batch_size=8, epochs=1):
    wandb.init(project="honours-sgtm")
    trainer_args = TrainingArguments(
        output_dir=out,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        save_steps=10000,
        report_to='wandb',
        prediction_loss_only=True,
    )
    trainer = Trainer(
        model=model,
        args=trainer_args,
        train_dataset=dataset,
    )
    trainer.train()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', required=True, type=Path)
    parser.add_argument('--tokens', required=True, type=Path)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--scratch_model', action='store_true')
    parser.add_argument('--type', default='text')
    parser.add_argument('--vocab_size', default=30522)

    args = parser.parse_args()
    tokenizer_type, vocab_size = re.findall('tokens-(scratch|pretrained)-([0-9]+).pkl$', str(args.tokens))[0]
    model_type = 'scratch' if args.scratch_model else 'pretrained'
    name = f'bert-{model_type}-{tokenizer_type}-{vocab_size}-{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")}'
    out_dir = args.out_dir / name
    out_dir.mkdir(exist_ok=True, parents=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.scratch_model:
        print("Using scratch model")
        model = BertForPreTraining(BertConfig())
    else:
        print("Using pretrained model")
        model = BertForPreTraining.from_pretrained('bert-base-uncased')

    if not args.scratch_model:
        if tokenizer_type == 'scratch':
            print(
                "Error: scratch_tokens cannot be used when using a pretrained model"
            )
            exit(1)
    else:
        if tokenizer_type == 'scratch':
            print("Using scratch tokenizer")
        else:
            print("Using pretrained tokenizer")
    f = open(args.tokens, 'rb')
    tokens = pickle.load(f)
    f.close()
    dataset = AgNewsDataset(tokens)
    
    train(
        out=out_dir,
        model=model,
        dataset=dataset,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )


if __name__ == "__main__":
    main()
# %%
