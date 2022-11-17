import random
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from transformers import DataCollatorForLanguageModeling


class BertDataset(Dataset):

    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        self.tokenizer_vocab = tokenizer.get_vocab()
        print('initied dataset')

    def __len__(self):
        return len(self.data)

    def tokenize(self, text: str):
        tokens = self.tokenizer.basic_tokenizer.tokenize(
            text,
            never_split=self.tokenizer_vocab.keys(),
        )
        tokens = self.tokenizer.encode_plus(tokens,
                                            max_length=512,
                                            trancation='longest',
                                            return_tensors='pt',
                                            return_special_tokens_mask=True,
                                            padding='max_length')

        return tokens

    def __getitem__(self, index):
        line = self.data[index]
        tokens = self.tokenize(line)
        # self.tokenizer.encode_plus()
        # tokens = self.tokenizer(line,
        #                         return_tensors='pt',
        #                         return_special_tokens_mask=True,
        #                         padding='max_length')
        return {
            "input_ids": tokens['input_ids'].squeeze(0)[:512],
            "attention_mask": tokens['attention_mask'].squeeze(0)[:512],
            "token_type_ids": tokens['token_type_ids'].squeeze(0)[:512],
            "special_tokens_mask":
            tokens['special_tokens_mask'].squeeze(0)[:512]
        }


class BertDataModule(pl.LightningDataModule):

    def __init__(self, data, tokenizer, batch_size=4, num_workers=4):
        super().__init__()
        random.shuffle(data)

        self.train_data = data
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.collate_fn = DataCollatorForLanguageModeling(self.tokenizer,
                                                          mlm=True,
                                                          mlm_probability=0.15)
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_dataset = BertDataset(self.train_data, self.tokenizer)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          self.batch_size,
                          collate_fn=self.collate_fn,
                          num_workers=self.num_workers,
                          persistent_workers=True)