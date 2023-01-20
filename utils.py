import os
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

def read_lndoc(path, non_def = True):
    if os.stat(path).st_size > 0:
        df = pd.read_csv(path, delimiter='\t', header=None)
        df = df.rename(columns={0:'texts'})
        name = path.split('/')[-1].split('.')[0]
        if non_def:
            df['labels'] = 1 if name in ['Y','2'] else 0
        else:
            df['labels'] = 1 if name in ['Y', 'N1', '2','1'] else 0
        df = df.drop_duplicates()
        return df

def reformat(path, non_def = True):
    not_data = read_lndoc(path + '/0.lndoc')
    def_data = read_lndoc(path + '/2.lndoc')
    if non_def:
        weak_data = read_lndoc(path + '/1.lndoc', True)
        not_data = pd.concat([not_data, weak_data], axis=0, ignore_index=True)
    else:
        weak_data = read_lndoc(path + '/1.lndoc', False)
        def_data = pd.concat([def_data, weak_data], axis=0, ignore_index=True)
        
    data = pd.concat([def_data, not_data], axis=0, ignore_index=True)
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
    # if non_def:
    #     df = df[df['labels'] != 1]
    # else:
    #     df = df[df['labels'] != 0]
    if non_def:
        df['labels'] = [1 if x == 2.0 else 0 for x in df['labels']]
    else:
        df['labels'] = [1 if x == 2.0 or x == 1.0 else 0 for x in df['labels']]
    df['texts'] = [' '.join(word_tokenize(x)) for x in df['texts']]
    return df