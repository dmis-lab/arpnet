# ARPNet
This repository provides implementation code of ARPNet, a Antidepressant Response Prediction Network for Major Depressive Disorder.

## Installation
The implemntation code of ARPNet was tested with Python2.7.12 and TensorFlow1.6.0. 
The experiments were conducted on a single TITAN Xp GPU machine which has 12GB of RAM.

## Datasets
We conducted experients on the data of 121 patients with MDD collected from Korea University Anam Hospital, Seoul, Korea. All the patients who are ethnically Korean were examined by trained psychiatrists using a structured clinical questionnaire. The depression severity of the patients was measured using the 17-item Hamilton Depression Rating(HAM-D17) scale and 21-item Hamilton Depression Rating (HAM-D 21) scale at every visit. We employ the HAM-D 17 scale for the predictions on the degree of the antidepressant response. The collected MDD patient data includes the demographic, genetic, and MRI information of only patients who have consented to the use of their data for this study. Some patients did not consent to the use of certain data, so the data available for each
patient may vary. If one patient’s i-th feature is missing, we randomly choose one of the i-th features of all the remaining patients, and use it as the i-th feature of the patient. Specifically, 67, 71, 96, and 91 patients
consented to the use of their demographic, MRI , and genetic information, respectively.

### Named Entity Recognition (NER)
Download and unpack the NER datasets provided above (**[`Named Entity Recognition`](http://gofile.me/6pN25/avQHrfPRf)**). From now on, `$NER_DIR` indicates a folder for a single dataset which should include `train_dev.tsv`, `train.tsv`, `devel.tsv` and `test.tsv`. For example, `export NER_DIR=~/bioBERT/biodatasets/NERdata/NCBI-disease`. Following command runs fine-tuining code on NER with default arguments.
```
mkdir /tmp/bioner/
python run_ner.py \
    --do_train=true \
    --do_eval=true \
    --vocab_file=$BIOBERT_DIR/vocab.txt \
    --bert_config_file=$BIOBERT_DIR/bert_config.json \
    --init_checkpoint=$BIOBERT_DIR/biobert_model.ckpt \
    --num_train_epochs=10.0 \
    --data_dir=$NER_DIR/ \
    --output_dir=/tmp/bioner/
```
You can change the arguments as you want. Once you have trained your model, you can use it in inference mode by using `--do_train=false --do_predict=true` for evaluating `test.tsv`.
The result will be printed as stdout format. For example, the result for NCBI-disease dataset will be like this:
```
INFO:tensorflow:***** TEST results *****
INFO:tensorflow:  eval_f = 0.9028707
INFO:tensorflow:  eval_precision = 0.8839457
INFO:tensorflow:  eval_recall = 0.92273223
INFO:tensorflow:  global_step = 2571
INFO:tensorflow:  loss = 25.894125
```
(tips : You should go up a few lines to find the result. It comes before `INFO:tensorflow:**** Trainable Variables ****` )

### Relation Extraction (RE)
Download and unpack the RE datasets provided above (**[`Relation Extraction`](http://gofile.me/6pN25/aT0oswqfr)**). From now on, `$RE_DIR` indicates a folder for a single dataset. `{TASKNAME}` means the name of task such as gad or euadr. For example, `export RE_DIR=~/bioBERT/biodatasets/REdata/GAD/1` and `--task_name=gad`. Following command runs fine-tuining code on RE with default arguments.
```
python run_re.py \
    --task_name={TASKNAME} \
    --do_train=true \
    --do_eval=true \
    --do_predict=true \
    --vocab_file=$BIOBERT_DIR/vocab.txt \
    --bert_config_file=$BIOBERT_DIR/bert_config.json \
    --init_checkpoint=$BIOBERT_DIR/biobert_model.ckpt \
    --max_seq_length=128 \
    --train_batch_size=32 \
    --learning_rate=2e-5 \
    --num_train_epochs=3.0 \
    --do_lower_case=false \
    --data_dir=$RE_DIR/ \
    --output_dir=/tmp/RE_output/ 
```
The predictions will be saved into a file called `test_results.tsv` in the `output_dir`. Once you have trained your model, you can use it in inference mode by using `--do_train=false --do_predict=true` for evaluating test.tsv. Use `./biocodes/re_eval.py` in `./biocodes/` folder for evaluation.
```
python ./biocodes/re_eval.py --output_path={output_dir}/test_results.tsv --answer_path=$RE_DIR/test.tsv
```
The result for GAD dataset will be like this:
```
.tsv
recall      : 92.88%
specificity : 67.19%
f1 score    : 83.52%
precision   : 75.87%
```
Please be aware that you have to move `output_dir` to make new model. As some RE datasets are 10-fold divided, you have to make different output directories to train a model with different datasets.

### Question Answering (QA)
To download QA datasets, you should register in [BioASQ website](http://participants-area.bioasq.org). After the registration, download **[`BioASQ Task B`](http://participants-area.bioasq.org/Tasks/A/getData/)** data, and unpack it to some directory `$BIOASQ_DIR`. Finally, download **[`Question Answering`](http://gofile.me/6pN25/C9iSvkPCr)**, our pre-processed version of BioASQ-4/5b datasets, and unpack it to `$BIOASQ_DIR`.

Pleas use `BioASQ-*.json` for training and testing the model. This is necessary as the input data format of BioBERT is different from BioASQ dataset format. Also, please be informed that the do_lower_case flag should be set as `--do_lower_case=False`. Following command runs fine-tuining code on QA with default arguments.
```
python run_qa.py \
     --do_train=True \
     --do_predict=True \
     --vocab_file=$BIOBERT_DIR/vocab.txt \
     --bert_config_file=$BIOBERT_DIR/bert_config.json \
     --init_checkpoint=$BIOBERT_DIR/biobert_model.ckpt \
     --max_seq_length=384 \
     --train_batch_size=12 \
     --learning_rate=3e-5 \
     --doc_stride=128 \
     --num_train_epochs=50.0 \
     --do_lower_case=False \
     --train_file=$BIOASQ_DIR/BioASQ-train-4b.json \
     --predict_file=$BIOASQ_DIR/BioASQ-test-4b-1.json \
     --output_dir=/tmp/QA_output/
```
The predictions will be saved into a file called `predictions.json` and `nbest_predictions.json` in the `output_dir`.
Run `transform_nbset2bioasqform.py` in `./biocodes/` folder to convert `nbest_predictions.json` to BioASQ JSON format, which will be used for the official evaluation.
```
python ./biocodes/transform_nbset2bioasqform.py --nbest_path={QA_output_dir}/nbest_predictions.json --output_path={output_dir}
```
This will generate `BioASQform_BioASQ-answer.json` in `{output_dir}`.
Clone **[`evaluation code`](https://github.com/BioASQ/Evaluation-Measures)** from BioASQ github and run evaluation code on `Evaluation-Measures` directory. Please note that you should always put 5 as parameter for -e.
```
cd Evaluation-Measures
java -Xmx10G -cp $CLASSPATH:./flat/BioASQEvaluation/dist/BioASQEvaluation.jar evaluation.EvaluatorTask1b -phaseB -e 5 \
    $BIOASQ_DIR/4B1_golden.json \
    RESULTS_PATH/BioASQform_BioASQ-answer.json
```
As our model is only on factoid questions, the result will be like
```
0.0 0.4358974358974359 0.6153846153846154 0.5072649572649572 0.0 0.0 0.0 0.0 0.0 0.0
```
where the second, third and fourth numbers will be SAcc, LAcc and MRR of factoid questions respectively.
Note that we pre-trained our model on SQuAD dataset to get the state-of-the-art performance. Please check our paper for details.

## License and Disclaimer
Please see LICENSE file for details. Downloading data indicates your acceptance of our disclaimer.

## Citation

If we submit the paper to a conference or journal, we will update the BibTeX.

## Contact information

For help or issues using ARPNet, please submit a GitHub issue. Please contact Buru Chang
(`buru_chag (at) korea.ac.kr`), or Yonghwa Choi (`yonghwachoi (at) korea.ac.kr`) for communication related to BioBERT.
