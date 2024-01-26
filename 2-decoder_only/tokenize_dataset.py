import pandas as pd
import string
import matplotlib.pyplot as plt
import seaborn as sns
import sentencepiece as spm
import re
import argparse



def get_data(path):
    df = pd.read_excel(path)
    return df

def clean(text):
    for p in string.punctuation:
        if p != '.':
            text = text.replace(p, '')
    text = re.sub("\n|\r|\t", " ",  text)
    text=' '.join(text.split(' '))
    text=text.replace(' .','.\n')
    text=text.lower()
    return text

def preprocess(df):
    # df.drop(['Unnamed: 0', 'section_names'], axis=1, inplace=True)
    df['article']=df['article'].apply(clean)
    df['abstract']=df['abstract'].apply(clean)
    return df 

def tokenize(df,path,model_name,vocab_size=32000):
    text = df['article'].tolist() ## make our article into a list of 
    text.extend(df['abstract'].tolist()) ## add the abstracts to the list
    with open(f'{path}', 'w') as fw:
        for l in text:
            fw.write(l)
    spm.SentencePieceTrainer.train(f'--input=data_ready_for_tokenization_spm.txt --model_type=bpe --model_prefix={model_name} --user_defined_symbols=<pad>,<EOS> --vocab_size={vocab_size}')

def main():
    # path = 'test_hugging_face_scientific.xlsx'
    # model= 'spm_v3'
    # preped_data = 'data_ready_for_tokenization_spm.txt'

    parser = argparse.ArgumentParser(description='pass path of the dataset,name of model, name of preped data for tokenization')
    parser.add_argument('--input', type=str, help='dataset in xlxs format')
    parser.add_argument('--output_model', type=str, help='model name')
    parser.add_argument('--output_txt', type=str, help='output txt file')

    args = parser.parse_args()

    # Access parsed arguments
    path = args.input
    model = args.output_model
    preped_data = args.output_txt
    df = get_data(path)
    df = preprocess(df)
    tokenize(df,preped_data,model,vocab_size=32000)

if __name__ == '__main__':
    main()  