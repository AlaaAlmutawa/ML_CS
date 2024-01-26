import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import (
    T5ForConditionalGeneration,
    T5TokenizerFast
)
from fine_tune_T5_dataset import PapersSummaryDataModule
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import sys
import time

pl.seed_everything(42)

## model prep
MODEL_NAME = 't5-base'
tokenizer = T5TokenizerFast.from_pretrained(MODEL_NAME)
N_EPOCHS = 10
BATCH_SIZE = 2

torch.cuda.empty_cache()

## Model training module
class PapersSummaryModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME, return_dict=True) # already includes the language modeling head

    def forward(self, input_ids, attention_mask, decoder_attention_mask, labels=None):
        output = self.model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask
        )

        return output.loss, output.logits

    def training_step(self, batch, batch_size):
        input_ids = batch['text_input_ids']
        attention_mask = batch['text_attention_mask']
        labels = batch['labels']
        labels_attention_mask = batch['labels_attention_mask']

        loss, outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=labels_attention_mask,
            labels=labels
        )

        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_size):
        input_ids = batch['text_input_ids']
        attention_mask = batch['text_attention_mask']
        labels = batch['labels']
        labels_attention_mask = batch['labels_attention_mask']

        loss, outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=labels_attention_mask,
            labels=labels
        )

        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_size):
        input_ids = batch['text_input_ids']
        attention_mask = batch['text_attention_mask']
        labels = batch['labels']
        labels_attention_mask = batch['labels_attention_mask']

        loss, outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=labels_attention_mask,
            labels=labels
        )

        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-5)
        return [optimizer]


def main (path):
    ## Data cleaning and split
    df = pd.read_excel(path)
    df.columns = ['article', 'summary']
    df = df.dropna()
    df.rename(columns={'article': 'text', 'abstract':'summary'}, inplace = True)
    train_df, val_df = train_test_split(df, test_size=0.2)
    test_df = val_df  # Inference is not done on this data split

    data_module = PapersSummaryDataModule(train_df, test_df, val_df, tokenizer, text_max_token_len = 2000, summary_max_token_len = 512)
    model = PapersSummaryModel()

    ts = time.time()
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',
        filename=f'best-checkpoint_{ts}',
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min'
    )

    logger = TensorBoardLogger("lightning_logs", name=f'papers-summary-new_{ts}')

    trainer = pl.Trainer(
        logger=logger,
        callbacks=[checkpoint_callback],
        max_epochs=N_EPOCHS
    )

    print('Start training')
    trainer.fit(model, data_module)
    print('End training')

if __name__ == '__main__':
    path = sys.argv[1]  ## input excel file path
    main(path) 

