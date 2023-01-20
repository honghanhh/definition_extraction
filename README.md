# Definition_extraction

## Introduction

Definition Extraction is an NLP task that automatically detects and identify the terms and their corresponding definition from the unstructured text sequences. In the scope of our research, we focus on the first subtask where the Definition Extraction can be formulated as a binary classification task to detect if it is a definitional sequence or not given the input of text sequences.

## Datasets

We propose a novel Slovene dataset for the evaluation of Definition Extraction tools (RSDO-def). The corpus was collected in the scope of the project Development of Slovene in a Digital Environment – Language Resources and Technologies. The description of the corpus can be found in the [readme.txt](./datasets/readme.txt).

## Model

To run the best Transformers-based approach, run the following command:

```bash
python binary_classifier.py --is_non_def True --model EMBEDDIA/sloberta --output_dir ./model/SloBERTa_Y_N --model_dir ./model/SloBERTa_Y_N_model --result_dir SloBERTa_Y_N_output.pkl

```

To reproduce the results of all the Transformers-based models we have experimented, run the following command:

```bash
chmod +x run.sh
./run.sh
```

## Results

The results can be found in the [results](./results) folder.

## Reference

The paper will be available soon.

## Contributors

- [@honghanhh](https://github.com/honghanhh)
