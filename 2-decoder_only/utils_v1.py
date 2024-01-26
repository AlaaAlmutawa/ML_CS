import pandas as pd
import string
import re
from datasets import load_dataset

def get_dataset(output_dir,split):
    scientific = load_dataset("scientific_papers", 'pubmed', split=split) ## 6658 rows. we can just use this
    scientific = scientific.to_pandas()
    scientific.drop(['section_names'], axis=1, inplace=True)
    scientific = scientific.dropna()
    scientific.to_excel(f"{output_dir}/scientific_papers_pubmed_{split}.xlsx",index=False)
    return scientific

def preprocess_util(row):
        # Convert all text to lowercase
    for p in string.punctuation:
        if p != '.': ## keep fullstop
            row = row.replace(p, '')
    row = row.lower()
    # Remove newlines and double spaces
    row = re.sub("\n|\r|\t", " ",  row)
    row = ' '.join(row.split(' '))

    return row

def preprocess(df):
    # Define the custom preprocessing function
    # Apply the preprocessing to the train and test datasets
    df['article'] = df['article'].apply(preprocess_util)
    df['abstract'] = df['abstract'].apply(preprocess_util)

    return df
