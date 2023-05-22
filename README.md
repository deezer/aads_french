# 📖 Automatic Annotation of Direct Speech in Written French Narratives

Automatic Annotation of Direct Speech (AADS) is the task of identifying Direct Speech (DS) text passages from narrative parts. It aims at separating what is uttered by characters inhabiting the storyworld from narrator's words.
This repository contains the code of **Annotation of Direct Speech in Written French Narratives**, accepted for publication to [ACL 2023](https://2023.aclweb.org/). 


## Task formalization

We consider AADS as a sequence labeling task that can be solved using literary conventions (regular expressions) or leveraging ML and DL methods trained on annotated novels.

Regular Expression methods serve as baselines for the experiments. Two schemes based on Language Models (LMs) and word-embeddings are adapted for the purpose of these experiments.
All models' performances are benchmark on several metrics, either on the whole labels set or aggregated per chapter.

## Implemented methods

We benchmark three methods : 
- Simple baselines are implemented with regular expression.
-  A BiLST-CRF deep learning architecture with to stacked Flair and FastText embeddings.
- Fine-tuned CamemBERT.

## Evaluation Schemes

We evaluate performances of each model using: 
- Token-level precision, recall, F1 scores
- Span-level precision, recall, F1 scores (Strict Sequence Match)
- Purity/Coverage 
- Zone Map Error

## Corpus

The experiments showcased in this repository rely on the compilation of four pre-existing corpora of excerpts of public domain french novels published during the XVIII and XIX century.

The *fr-litbank* and *sini-bonus* corpora are used as is, *SynPaFlex* and *Straight Talk!* corpora were reannotated to fit the chosen framework (ie. annotation of whole chapters with DS labels at the token level).

*fr-litbank*, *sini-bonus* and *SynPaFlex* are considered as the `main` corpus, while *Strait Talk!* files are considered as `noisy` corpus due to its poorer formating quality (lack of breaklines, misplaced breaklines, etc.).


Main corpus statistics are shown in the following table:

|   |Files|Tokens|Sentences|Av. DS%|
|---:|:---:|:---:|:---:|:---:|
|Main — *train*|37|353,658|13,881 |40 (30)|
|Main — *val*|6|71,739|3,443|35 (21)|
|Main — *test*|6|62,825|2,303|30 (22)|
|Noisy — *ST!*|38|236,645|10,020|38 (21)|
|**Total**|**87**|**724,867**|**29,647**|**38 (25)**|

We provide the raw annotated corpus in json format, available at `data/aggregated_corpus.json`. We also provide tokenized  versions of the corpus in `.tsv` format in the subsequent subfolders. Note that we also split the `main` and `noisy` corpus in train/val/test splits. These files were generated using the following command line: 
```bash
python preprocess/tokenize_corpus.py --data_dir /src/data/aggregated_corpus.json --output_dir src/data/ --tokenizer [$TOKENIZER]
```
where the tokenizer variable can be set to `spacy_tokenization` or `nltk_tokenization`. The `spacy_tokenization` was used for all experiments. 

# 🛠 Reproducing results

## Installation

This repository requires python $\geq$ 3.8.3. Install dependancies in a fresh virtual environment with: 

```bash 
pip install -r requirements.txt
```

Install spacy tokenizer `fr_core_news_sm` with the following command:
```bash
python -m spacy download fr_core_news_sm
```

## Runing the benchmark

The following line will train the ML models and compute the performances on both the validation set and the test set:

```bash
python run_experiments.py --configs_folder experiments_configs/best_configs/main
```

Then, next line can be ran, leveraging the previously trained model on noisy corpus:
```bash
python run_experiments.py --configs_folder experiments_configs/best_configs/noisy
```

Models' performance scores are stored as `.json` files in the [`results`](./results) directory following the convention: `model_type/tokenization_type/model_name/corpus_type/`. Performances on the validation split are stored as `val_performances.json` and performances on the test split as `test_performances.json`.



# Predicting Direct Speech

The [`prediction` folder](./prediction) presents how to leverage implemented and trained models to predict Direct Speech on unseen texts.
The [demo_prediction](./prediction/demo_prediction.ipynb) notebook shows steps to preprocess raw text files and predict direct speech.



# Clause-consistent predictions

We ran a simple post-processing experiment based on clausse-consistent predictions. We smoothe predictions by clause (through majority vote) to improve the models' performances.
Based on prediction `.tsv` files, smooth predictions are obtained using the following command line: 

```bash
python post_processing.py --result_path [$RESULT_PATH]
```
where the variable `RESULT_PATH` points to the folder containing stored prediction in `.tsv` format. 

---

# Cite 

If you happen to use or modify this code, please remember to cite our paper: 

```
@InProceedings{durandard_directspeech2023,
    title = {Automatic Annotation of Direct Speech in French Narratives},
    booktitle = {Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
    author = {Durandard, Noé and
      Tran, Viet Anh and
      Michel, Gaspard and
      Epure, V. Elena},
    year = {2023}
}
```

---
