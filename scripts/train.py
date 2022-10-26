import os
from random import seed
import shutil
import torch
import pytorch_lightning as pl
import argparse
from transformers import BertForMaskedLM, BertConfig, Trainer, TrainingArguments
from utils.dataset import BertDataModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger
from tokenizers import AddedToken
from utils.io import load_seed, load_tokenizer, load_vocab
from rich.logging import RichHandler
import logging
import wandb


class BertTrainingModule(pl.LightningModule):

    def __init__(self, model, from_sratch=False, learning_rate=5e-5):
        super().__init__()
        self.model = model
        self.from_scratch = from_sratch
        self.learning_rate = learning_rate

    def training_step(self, batch, batch_idx):
        res = self.model(**batch)
        loss = res.loss
        self.log('train_loss',
                 loss,
                 on_epoch=True,
                 on_step=False,
                 sync_dist=True)
        return loss

    def configure_optimizers(self):
        lr = self.learning_rate
        milestones = [10, 25, 50, 75, 100, 125]
        if self.from_scratch:
            milestones = [25, 50, 75, 100, 125]
        optim = torch.optim.AdamW(self.parameters(), lr=lr)
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.MultiStepLR(
                optim,
                milestones,
            ),
            "interval": "epoch",
        }
        return [optim], [scheduler]


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path',
                        type=str,
                        required=True,
                        help='Path to Train-Labeled')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument("--vocab", type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--tokenizer', type=str, default=None)
    parser.add_argument("--verbose", "-v", action='store_true')
    parser.add_argument("--scratch", action='store_true')
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--upload_model", action='store_true')
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    return args


def main(args):
    tokenizer = load_tokenizer(args.tokenizer)
    vocab = load_vocab(args.vocab)
    model = None
    if args.tokenizer is not None and args.tokenizer.strip() != '':
        logging.info(
            "Training BERT from scratch because '--tokenizer' is specified")
        config = BertConfig(vocab_size=len(tokenizer.vocab))
        model = BertForMaskedLM(config)
    elif args.scratch:
        logging.info(
            "Training BERT from scratch because '--scratch' is specified")
        config = BertConfig()
        model = BertForMaskedLM(config)
    else:
        logging.info("Fine tunning BERT")
        model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    seeds = load_seed(args.seeds, combine_result=True)
    new_tokens = list(set(vocab) - set(tokenizer.vocab.keys()))
    new_tokens = [
        AddedToken(word, single_word=True) for word in new_tokens
        if '_' in word
    ]
    seed_tokens = [AddedToken(word, single_word=True) for word in seeds]
    tokenizer.add_tokens(new_tokens)
    tokenizer.add_tokens(seed_tokens)
    # logging.info(f"adding {new_tokens} to tokenizer")
    logging.info(f"new tokenizer now has shape {len(tokenizer)}")
    tokenizer_dir = os.path.join(args.output_dir, "tokenizer")
    model_dir = os.path.join(args.output_dir, f"finetuned-model")
    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer.save_pretrained(tokenizer_dir)
    logging.info(f"save tokenizer to {tokenizer_dir}")
    model.resize_token_embeddings(len(tokenizer))
    os.makedirs(args.output_dir, exist_ok=True)
    checkpointer = ModelCheckpoint(
        dirpath=args.output_dir,
        every_n_epochs=10,
    )
    data = None
    with open(args.dataset_path, 'r') as f:
        data = f.readlines()

    data_module = BertDataModule(
        data,
        tokenizer,
        args.batch_size,
        args.num_workers,
    )
    training_module = BertTrainingModule(model, from_sratch=args.scratch)
    trainer: pl.Trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[checkpointer],
        logger=WandbLogger(project="honours-sgtm"),
        strategy="ddp_find_unused_parameters_false",
        auto_scale_batch_size="binsearch",
    )
    # trainer.tune(training_module, datamodule=data_module)
    trainer.fit(training_module, data_module)
    hg_trainer = Trainer(model=model)
    logging.info(f"Saving model and config to {model_dir}")
    hg_trainer.save_model(model_dir)
    if args.upload_model:
        logging.info(
            f"zipping tokenizer and model to 'model-{trainer.logger.name}.zip'"
        )
        shutil.make_archive(f"model-{trainer.logger.name}", "zip",
                            args.output_dir)
    return f'model-{trainer.logger.name}.zip'


if __name__ == '__main__':
    args = parser_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO, handlers=[RichHandler()])
    else:
        logging.basicConfig(handlers=[RichHandler()])
    zip_name = main(args)
    if args.upload_model:
        logging.info(f"Uploading {zip_name} to wandb")
        wandb.save(zip_name)