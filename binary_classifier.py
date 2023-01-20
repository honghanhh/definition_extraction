import os
os.environ["CUDA_VISIBLE_DEVICES"]='4'
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
    parser.add_argument('--is_non_def', type=bool, default=True, help='Use non-definitional data or not.')
    parser.add_argument('--model', type=str, default='EMBEDDIA/sloberta', help='Model to use.')
    parser.add_argument('--output_dir', type=str, default='./SloBERTa_Y_N', help='Output directory.')
    parser.add_argument('--model_dir', type=str, default='./SloBERTa_Y_N_model', help='Model directory.')
    parser.add_argument('--result_dir', type=str, default='./SloBERTa_Y_N_output.pkl', help='Result directory.')
    

    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
        
    path = './classifier_18_20_22/corpus/'
    if args.is_non_def:
        not_df = read_lndoc(path + 'sl/DF_NDF_wiki/N.lndoc')
        rand_df = preprocess_raw_corpus(path + 'Definitions_5-200_ZRC_ocena.csv', 
                                    'sentence', 
                                    'SKUPNA OCENA PRIMERNOSTI', 
                                    'csv', True)
        pattern_df = preprocess_raw_corpus(path + 'SentEx_results_patterns_merged_ZRC.xlsx',
                                        'sentence',
                                        'Ocena (glede na termine, kjer bi jim lahko to pripisali)',
                                        'xlsx', True)
    else:
        not_df = read_lndoc(path + 'sl/DF_NDF_wiki/N1.lndoc')
        rand_df = preprocess_raw_corpus(path + 'Definitions_5-200_ZRC_ocena.csv', 
                                    'sentence', 
                                    'SKUPNA OCENA PRIMERNOSTI', 
                                    'csv', False)
        pattern_df = preprocess_raw_corpus(path + 'SentEx_results_patterns_merged_ZRC.xlsx',
                                        'sentence',
                                        'Ocena (glede na termine, kjer bi jim lahko to pripisali)',
                                        'xlsx', False)

    def_df = read_lndoc(path + 'sl/DF_NDF_wiki/Y.lndoc')
    data = pd.concat([def_df, not_df])
    data['texts'] = [' '.join(word_tokenize(x)) for x in data['texts']]
    

    X_train, X_test, y_train, y_test = train_test_split(data[['texts']],
                                                        data['labels'],
                                                        test_size=0.25, 
                                                        shuffle = True,
                                                        random_state=42)

    X_train['labels'], X_test['labels'] = y_train, y_test
    train_df, test_df = Dataset.from_dict(X_train), Dataset.from_dict(X_test)
    

    all_files = [file for file in glob.glob("./classifier_18_20_22/corpus/sl/rsdo5*")]
    outputs =[]
    names = []
    for file in all_files:
        if args.is_non_def:
            outputs.append(reformat(file, True))
        else:
            outputs.append(reformat(file, False))
        names.append(file.split('/')[-1])

    outputs_dict = [Dataset.from_dict(x) for x in outputs]    

    data = dict(zip(names,outputs_dict))


    data['train'] = train_df
    data['test'] = test_df
    data['random_sampling'] = Dataset.from_dict(rand_df)
    data['pattern'] = Dataset.from_dict(pattern_df)
    data['all_rsdo'] = Dataset.from_dict(pd.concat(outputs, axis=0, ignore_index=True))
    raw_datasets = DatasetDict(data)
    
    class_weights = compute_class_weight(class_weight = 'balanced', classes = np.unique(y_train), y = y_train)
    class_weights =  [np.float32(x) for x in class_weights.tolist()]

    
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
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
