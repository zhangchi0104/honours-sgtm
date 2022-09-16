import os
import torch
import pytorch_lightning as pl
import argparse
from transformers import BertForMaskedLM
from utils.dataset import BertDataModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger

from utils.io import load_tokenizer


class BertTrainingModule(pl.LightningModule):

    def __init__(self, model):
        super().__init__()
        self.model = model

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
        optim = torch.optim.Adam(self.parameters(), lr=1e-4)
        scheduler = {
            "scheduler":
            torch.optim.lr_scheduler.MultiStepLR(
                optim,
                [10, 25, 50, 75, 100, 125],
            ),
            "interval":
            "epoch",
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
    parser = pl.Trainer.add_argparse_args(parser)
    return parser.parse_args()


def main(args):
    tokenizer = load_tokenizer(args.tokenizer)
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
    training_module = BertTrainingModule(model)
    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[checkpointer],
        logger=WandbLogger(project="honours-sgtm"),
    )
    trainer.fit(training_module, data_module)
    torch.save(
        model.model.state_dict(),
        os.path.join(args.output_dir, 'model_final.pth'),
    )


if __name__ == '__main__':
    args = parser_args()
    main(args)