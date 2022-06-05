from pathlib import Path

import torch.cuda
from transformers import BertForPreTraining, DataCollatorWithPadding , Trainer, TrainingArguments, BertConfig
from tokenizers import BertWordPieceTokenizer
from datasets import concatenate_datasets, load_dataset, Dataset
from transformers import BertTokenizer
import wandb
import datetime, argparse


def train(out, dataset, model, tokenizer, batch_size=8, epochs=1):
    wandb.init(project="honours-sgtm")
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,
    )
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
        data_collator=data_collator,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )
    trainer.train()

def main():
    now_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', required=True, type=Path)
    parser.add_argument('--file', required=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--scratch_model', action='store_true')
    parser.add_argument('--type', default='text')
    parser.add_argument('--scratch_tokenizer', action='store_true')
    parser.add_argument('--vocab_size', default=30522)

    args = parser.parse_args()
    tokenizer_type = 'pretrained' if not args.scratch_tokenizer else 'scratch'
    model_type = 'scratch' if args.scratch_model else 'pretrained'
    name = f'bert-{model_type}-{tokenizer_type}-{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")}'
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
        if args.scratch_tokenizer:
            print(
                "Warnning: --scratch_tokenizer is ignored when using pretrained model"
            )
        print('Using pretrained tokenizer')
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    else:
        if args.srcratch_tokenizer:
            print("Using scratch tokenizer")
            tokenizer = BertWordPieceTokenizer(
                clean_text=True,
                handle_chinese_chars=True,
                strip_accents=True,
                lowercase=True,
            )
            tokenizer.train(
                [args.file],
                vocab_size=args.vocab_size,
                min_frequency=2,
                show_progress=True,
                special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
                limit_alphabet=1000,
                wordpieces_prefix="##",
            )
        else:
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    dataset = load_dataset('ag_news')
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    dataset = dataset.map(tokenize_function, batched=True, num_proc=12)
    dataset.set_format("torch")
    dataset['all'] = concatenate_datasets([dataset['train'], dataset['test']])


    train(
        out=out_dir,
        model=model,
        tokenizer=tokenizer,
        dataset=dataset['all'].remove_columns(['label']),
        batch_size=args.batch_size,
        epochs=args.epochs,
    )


if __name__ == "__main__":
    main()
# %%
