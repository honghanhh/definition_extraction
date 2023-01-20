python binary_classifier.py --is_non_def True --model EMBEDDIA/sloberta                          --output_dir ./model/SloBERTa_Y_NN1    --model_dir ./model/SloBERTa_Y_NN1_model    --result_dir SloBERTa_Y_NN1_output.pkl --statistics SloBERTa_Y_NN1.csv

python binary_classifier.py --is_non_def True --model xlm-roberta-base                           --output_dir ./model/xlmr_Y_NN1        --model_dir ./model/xlmr_Y_NN1_model        --result_dir xlmr_Y_NN1_output.pkl      --statistics xlmr_Y_NN1.csv

python binary_classifier.py --is_non_def True --model bert-base-multilingual-cased               --output_dir ./model/bert_Y_NN1        --model_dir ./model/bert_Y_NN1_model        --result_dir bert_Y_NN1_output.pkl      --statistics bert_Y_NN1.csv

python binary_classifier.py --is_non_def True --model distilbert-base-multilingual-cased         --output_dir ./model/distilbert_Y_NN1  --model_dir ./model/distilbert_Y_NN1_model  --result_dir distilbert_Y_NN1_output.pkl --statistics distilbert_Y_NN1.csv

python binary_classifier.py --is_non_def True --model roberta                                    --output_dir ./model/roberta_Y_NN1      --model_dir ./model/roberta_Y_NN1_model    --result_dir roberta_Y_NN1_output.pkl    --statistics roberta_Y_NN1.csv


python binary_classifier.py --is_non_def False --model EMBEDDIA/sloberta                         --output_dir ./model/SloBERTa_YN1_N    --model_dir ./model/SloBERTa_YN1_N_model    --result_dir SloBERTa_YN1_N_output.pkl  --statistics SloBERTa_YN1_N.csv

python binary_classifier.py --is_non_def False --model xlm-roberta-base                          --output_dir ./model/xlmr_YN1_N        --model_dir ./model/xlmr_YN1_N_model        --result_dir xlmr_YN1_N_output.pkl       --statistics xlmr_YN1_N.csv

python binary_classifier.py --is_non_def False --model bert-base-multilingual-cased              --output_dir ./model/bert_YN1_N        --model_dir ./model/bert_YN1_N_model       --result_dir bert_YN1_N_output.pkl       --statistics bert_YN1_N.csv

python binary_classifier.py --is_non_def False --model distilbert-base-multilingual-cased        --output_dir ./model/distilbert_YN1_N  --model_dir ./model/distilbert_YN1_N_model  --result_dir distilbert_YN1_N_output.pkl  --statistics distilbert_YN1_N.csv

python binary_classifier.py --is_non_def False --model roberta                                   --output_dir ./model/roberta_YN1_N     --model_dir ./model/roberta_YN1_N_model     --result_dir roberta_YN1_N_output.pkl     --statistics roberta_YN1_N.csv



