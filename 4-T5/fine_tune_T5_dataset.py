## Dataset module (Can access train, test, validation splits easily)

import pandas as pd
from transformers import (
    T5TokenizerFast
)
from fine_tune_T5_data_prep import PapersSummaryDataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl
pl.seed_everything(42)

class PapersSummaryDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        val_df: pd.DataFrame,
        tokenizer: T5TokenizerFast,
        batch_size: int = 2,
        text_max_token_len: int = 3000,
        summary_max_token_len: int = 512
    ):
        super().__init__()

        self.train_df = train_df
        self.test_df = test_df
        self.val_df = val_df
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.text_max_token_len = text_max_token_len
        self.summary_max_token_len = summary_max_token_len

    def setup(self, stage=None):
        self.train_dataset = PapersSummaryDataset(
            self.train_df,
            self.tokenizer,
            self.text_max_token_len,
            self.summary_max_token_len
        )
        self.val_dataset = PapersSummaryDataset(
            self.val_df,
            self.tokenizer,
            self.text_max_token_len,
            self.summary_max_token_len
        )
        self.test_dataset = PapersSummaryDataset(
            self.test_df,
            self.tokenizer,
            self.text_max_token_len,
            self.summary_max_token_len
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2
        )