import random
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from transformers import DataCollatorForLanguageModeling


class BertDataset(Dataset):

    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        line = self.data[index]
        tokens = self.tokenizer(line,
                                return_tensors='pt',
                                return_special_tokens_mask=True,
                                max_length=512,
                                truncation=True,
                                padding='max_length')
        return {
            "input_ids": tokens['input_ids'].squeeze(0),
            "attention_mask": tokens['attention_mask'].squeeze(0),
            "token_type_ids": tokens['token_type_ids'].squeeze(0),
            "special_tokens_mask": tokens['special_tokens_mask'].squeeze(0)
        }


class BertDataModule(pl.LightningDataModule):

    def __init__(self,
                 data,
                 tokenizer,
                 train_ratio=0.8,
                 batch_size=4,
                 num_workers=4):
        super().__init__()
        random.shuffle(data)
        train_size = int(len(data) * train_ratio)
        self.train_data = data[:train_size]
        self.val_data = data[train_size:]
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.collate_fn = DataCollatorForLanguageModeling(self.tokenizer,
                                                          mlm=True,
                                                          mlm_probability=0.15)
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_dataset = BertDataset(self.train_data, self.tokenizer)
        self.val_dataset = BertDataset(self.val_data, self.tokenizer)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          self.batch_size,
                          collate_fn=self.collate_fn,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          self.batch_size,
                          collate_fn=self.collate_fn,
                          num_workers=self.num_workers)
