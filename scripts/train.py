import os
import torch
import pytorch_lightning as pl
import argparse
from transformers import BertForMaskedLM, BertConfig, Trainer, TrainingArguments
from utils.dataset import BertDataModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger

from utils.io import load_tokenizer
from rich.logging import RichHandler
import logging
import wandb


class BertTrainingModule(pl.LightningModule):

    def __init__(self, model, from_sratch=False):
        super().__init__()
        self.model = model
        self.from_scratch = from_sratch

    def training_step(self, batch, batch_idx):
        res = self.model(**batch)
        loss = res.loss
        self.log('train_loss',
                 loss,
                 on_epoch=True,
                 on_step=False,
                 sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        res = self.model(**batch)
        loss = res.loss
        self.log('val_loss',
                 loss,
                 on_epoch=True,
                 on_step=False,
                 sync_dist=True)
        return loss

    def configure_optimizers(self):
        lr = 1e-4
        milestones = [10, 25, 50, 75, 100, 125]
        if self.from_scratch:
            lr = 1e-2
            milestones = [25, 50, 75, 100, 125]
        optim = torch.optim.SGD(self.parameters(), lr=lr)
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
    parser.add_argument('--train_ratio',
                        type=float,
                        default=0.8,
                        help="ratio for the trainning set")
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--tokenizer', type=str, default='')
    parser.add_argument("--verbose", "-v", action='store_true')
    parser.add_argument("--scratch", action='store_true')
    parser.add_argument("--upload_model", action='store_true')
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    return args


def main(args):
    tokenizer = load_tokenizer(args.tokenizer)
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
        args.train_ratio,
        args.batch_size,
        args.num_workers,
    )
    training_module = BertTrainingModule(model, from_sratch=args.scratch)
    trainer: pl.Trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[checkpointer],
        logger=WandbLogger(project="honours-sgtm"),
        strategy="ddp_find_unused_parameters_false",
    )
    trainer.fit(training_module, data_module)
    hg_trainer = Trainer(model=model)
    hg_trainer.save_model(args.output_dir)


if __name__ == '__main__':
    args = parser_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO, handlers=[RichHandler()])
    else:
        logging.basicConfig(handlers=[RichHandler()])
    main(args)