import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import sentencepiece as spm
import utils_v1
import trax
from trax import layers as tl
from trax.fastmath import numpy as jnp
import simple_gpt_model


## Refernece: 
## This was completed as a part of following tutorial provided by Coursera Deep Learning Specialization
## https://www.coursera.org/specializations/natural-language-processing
## https://github.com/LaurentVeyssier/TRAX_transformer_abstractive_summarization_model/blob/main/TRAX_transformer_summarizer_model.ipynb

## helper function to generate data 
def data_generator(data):
    for text, label in data:
        yield text, label

## tokenize an input using the output_model from the tokenization step. 
def tokenize(input_str, EOS=4):
    """Input str to features dict, ready for inference"""
    inputs =  next(trax.data.tokenize(iter([input_str]),
                                      vocab_dir='vocab_dir/',
                                      vocab_file='spm_v4.model', ## change to the tokenization model name
                                      vocab_type='sentencepiece'))

    return list(inputs) + [EOS]

## detokenize an input sequence 
def detokenize(integers):
    """List of ints to str"""

    s = trax.data.detokenize(integers,
                             vocab_dir='vocab_dir/',
                            vocab_file='spm_v4.model', ## change to the tokenization model name
                            vocab_type='sentencepiece')

    return wrapper.fill(s)

def main():
    df = utils_v1.get_dataset('data','test')
    df = utils_v1.preprocess(df)
    train, test = train_test_split(df, test_size=0.2)

    prepared_data = [(article, abstract) for article, abstract in zip(train['article'], train['abstract'])]
    val_prepared_data = [(article, abstract) for article, abstract in zip(test['article'], test['abstract'])]


    # Create a generator from your prepared data
    train_data_stream = data_generator(prepared_data)
    val_data_stream = data_generator(val_prepared_data)

    sp = spm.SentencePieceProcessor()
    sp.Load("vocab_dir/spm_v4.model")  # Load the trained tokenization model ## change to the tokenization model name

    pad_id = sp.piece_to_id('<pad>')  # Get the ID of the <pad> token

    eos_id = sp.piece_to_id('<EOS>')  # Get the ID of the <pad> token

    # Special tokens
    SEP = pad_id # Padding or separator token
    EOS = eos_id # End of sentence token

    # preprocess helper function
    def preprocess(ds):
        for ( article , abstract ) in ds:
            joint = np.array(list(article) + [EOS,SEP]+ list(abstract) + [EOS])
            mask = [0] * (len(list(article)) + 2) + [1] * (len(list(abstract)) + 1) # Accounting for EOS and SEP
            yield joint, joint, np.array(mask)


    input_pipeline = trax.data.Serial(
        trax.data.Tokenize(vocab_dir='vocab_dir/', ## change to the directory name where the tokenization model is 
                        vocab_file='spm_v4.model', ## change to the tokenization model name
                        vocab_type='sentencepiece'),
        # Uses function defined above
        trax.data.TruncateToLength(len_map={0: (1024,), 1: (1024,)}),
        preprocess
    )

    # Apply preprocessing to data streams.
    train_stream = input_pipeline(train_data_stream)
    eval_stream = input_pipeline(val_data_stream)

    train_input, train_target, train_mask = next(train_stream)

    assert sum((train_input - train_target)**2) == 0 

    boundaries =  [128, 256,  512, 1024]
    batch_sizes = [16,    8,    4,    2, 1]

    # Create the streams.
    train_batch_stream = trax.data.BucketByLength(
        boundaries, batch_sizes)(train_stream)

    eval_batch_stream = trax.data.BucketByLength(
        boundaries, batch_sizes)(eval_stream)
    
    input_batch, _, mask_batch = next(train_batch_stream)
    
    loop = simple_gpt_model.training_loop(simple_gpt_model.TransformerLM, train_batch_stream, eval_batch_stream)
    loop.run(260)

if __name__ == '__main__':
    main() 




