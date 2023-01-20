import os
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

def read_lndoc(path, ):
    if os.stat(path).st_size > 0:
        df = pd.read_csv(path, delimiter='\t', header=None)
        df = df.rename(columns={0:'texts'})
        df['labels'] = 1 if path.split('/')[-1].split('.')[0] == 'Y' or path.split('/')[-1].split('.')[0] =='2' else 0
        df = df.drop_duplicates()
        return df

def reformat(path, non_def = True):
    if non_def:
        not_data = read_lndoc(path + '/0.lndoc')
    else:
        not_data = read_lndoc(path + '/1.lndoc')
    def_data = read_lndoc(path + '/2.lndoc')
    data = pd.concat([def_data, not_data])
    data['texts'] = [' '.join(word_tokenize(x)) for x in data['texts']]
    return data


def preprocess_raw_corpus(path, col1, col2, file='csv', non_def = True):
    if file == 'csv':
        df = pd.read_csv(path)[[col1, col2]]
    else:
        df = pd.read_excel(path)[[col1, col2]]
    
    df = df.rename(columns = {col1:'texts',
                            col2:'labels'})
    df = df.dropna(subset=['labels'])
    if non_def:
        df = df[df['labels'] != 1]
    else:
        df = df[df['labels'] != 0]
    df['labels'] = [1 if x == 2.0 else 0 for x in df['labels']]
    df['texts'] = [' '.join(word_tokenize(x)) for x in df['texts']]
    return df
