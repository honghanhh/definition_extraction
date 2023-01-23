import os
os.environ["CUDA_VISIBLE_DEVICES"]='5'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"
import glob
import pickle
import warnings
warnings.filterwarnings('ignore')
import argparse
import random
random.seed(3407)
import numpy as np
np.random.seed(3407)
import torch  
torch.manual_seed(3407)
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from datasets import  Dataset, DatasetDict
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report

from utils import *


def preprocess_function(examples):
    return tokenizer(examples["texts"], truncation=True)

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels").to(device)
        # forward pass
        outputs = model(**inputs)#.to(device)
        logits = outputs.get('logits')#.to(device)
        # compute custom loss
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor(class_weights)).to(device)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1)).to(device)
        return (loss, outputs) if return_outputs else loss


def pred_rsdo(df, trainer):
    test_predictions, test_labels, _ = trainer.predict(df)
    test_predictions = np.argmax(test_predictions, axis=1)
    return classification_report(test_labels, test_predictions, digits=4, output_dict=True)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Binary classifier.')
    parser.add_argument('--is_non_def', type=int, default= 1 , help='Use non-definitional data or not.')
    parser.add_argument('--model', type=str, default='EMBEDDIA/sloberta', help='Model to use.')
    parser.add_argument('--output_dir', type=str, default='./SloBERTa_Y_N', help='Output directory.')
    parser.add_argument('--model_dir', type=str, default='./SloBERTa_Y_N_model', help='Model directory.')
    parser.add_argument('--result_dir', type=str, default='./SloBERTa_Y_N_output.pkl', help='Result directory.')
    parser.add_argument('--statistics', type=str, default='./SloBERTa_Y_N_stats.csv', help='Stats.')


    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
        
    path = './classifier_18_20_22/corpus/'
    print(args)
    

    not_df = read_lndoc(path + 'sl/DF_NDF_wiki/N.lndoc')
    def_df = read_lndoc(path + 'sl/DF_NDF_wiki/Y.lndoc')

    if args.is_non_def == 1:
        weak_df = read_lndoc(path + 'sl/DF_NDF_wiki/N1.lndoc', 1)
        not_df = pd.concat([not_df, weak_df], axis=0, ignore_index=True)
        
        rand_df = preprocess_raw_corpus(path + 'Definitions_5-200_ZRC_ocena.csv', 
                                    'sentence', 
                                    'SKUPNA OCENA PRIMERNOSTI', 
                                    'csv', 1)
        pattern_df = preprocess_raw_corpus(path + 'SentEx_results_patterns_merged_ZRC.xlsx',
                                        'sentence',
                                        'Ocena (glede na termine, kjer bi jim lahko to pripisali)',
                                        'xlsx', 1)
    else:
        weak_df = read_lndoc(path + 'sl/DF_NDF_wiki/N1.lndoc', 0)
        def_df = pd.concat([def_df, weak_df], axis=0, ignore_index=True)
        
        rand_df = preprocess_raw_corpus(path + 'Definitions_5-200_ZRC_ocena.csv', 
                                    'sentence', 
                                    'SKUPNA OCENA PRIMERNOSTI', 
                                    'csv', 0)
        pattern_df = preprocess_raw_corpus(path + 'SentEx_results_patterns_merged_ZRC.xlsx',
                                        'sentence',
                                        'Ocena (glede na termine, kjer bi jim lahko to pripisali)',
                                        'xlsx', 0)
    
    data = pd.concat([def_df, not_df])
    
    wiki = get_value_counts(data, 'wiki')
    rand = get_value_counts(rand_df, 'rand_df')
    pat = get_value_counts(pattern_df, 'pattern_df')
    stats = wiki.merge(rand,on='labels', how='inner').merge(pat, on='labels', how='inner')
    # print(stats)
    
    data['texts'] = [' '.join(word_tokenize(x)) for x in data['texts']]
    

    X_train, X_test, y_train, y_test = train_test_split(data[['texts']],
                                                        data['labels'],
                                                        test_size=0.25, 
                                                        shuffle = True,
                                                        stratify = data['labels'],
                                                        random_state=42)

    X_train['labels'], X_test['labels'] = y_train, y_test
    train_df, test_df = Dataset.from_dict(X_train), Dataset.from_dict(X_test)
    
    all_files = [file for file in glob.glob("./classifier_18_20_22/corpus/sl/rsdo5*")]
    outputs =[]
    names = []
    for file in all_files:
        if args.is_non_def == 1:
            outputs.append(reformat(file, 1))
        else:
            outputs.append(reformat(file, 0))
        names.append(file.split('/')[-1])

    outputs_dict = [Dataset.from_dict(x) for x in outputs]     
    
    ### Just for checking data split
    all_rsdo = pd.concat(outputs, axis=0, ignore_index=True)
    rsdo_stats = get_value_counts(all_rsdo, 'rsdo')
    stats = rsdo_stats.merge(stats, on='labels', how='inner')
    stats.to_csv(args.statistics, index=False)
    print(stats)
     ### End checking

    data = dict(zip(names,outputs_dict))
 
    data['train'] = train_df
    data['test'] = test_df
    data['random_sampling'] = Dataset.from_dict(rand_df)
    data['pattern'] = Dataset.from_dict(pattern_df)
    data['all_rsdo'] = Dataset.from_dict(all_rsdo)
    raw_datasets = DatasetDict(data)
    
    class_weights = compute_class_weight(class_weight = 'balanced', classes = np.unique(y_train), y = y_train)
    class_weights =  [np.float32(x) for x in class_weights.tolist()]

    tokenizer = AutoTokenizer.from_pretrained(args.model,
                                            #   model_max_length=512,
                                              use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2).to(device)

    tokenized_data = raw_datasets.map(preprocess_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir= args.output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=10,
        weight_decay=0.01,
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(args.model_dir)

    
    results = []
    for name in ['test','random_sampling','pattern','all_rsdo']:
        metrics = pred_rsdo(tokenized_data[name], trainer)
        results.append({name: metrics})


    with open(args.result_dir, 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
