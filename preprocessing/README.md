# 🗂 Preprocessing

This module contains various utils methods and classes. It is among other things used in the first place to load the raw `json` corpus and generate `.tsv` files of tokenized text on which ML models rely (to generate the files directly, [jump here](#generating-files), or walk through the [demo notebook](./demo_gene.ipynb)).


<blockquote>

**Table of Content**

1. [Files](#files)
3. [Data Structure](#data-structure)
   1. [Raw Corpus](#raw-corpus)
   2. [Custom data classes](#custom-data-classes)
2. [Tokenizers](#tokenizers)
   1. [`Document`](#document)
   2. [`Token`](#token)
   3. [`nltk_tokenizer_pipe`](#nltk_tokenizer_pipe)
3. [Loading corpus](#loading-corpus)
4. [Generating Files](#generating-files)

</blockquote>

---


# Files

- [`nltk_tokenizer.py`](.nltk_tokenizer.py)
  - implementation of document tokenizer (sentence splitting + word tokenization) based on regular expressions implemented in the NLTK library.
- [`data_utils.py`](.data_utils.py)
  - data structure classes definitions as well as main methods to load corpus and generate `.tsv` splits.
- [`utils_converters.py`](.utils_converters.py)
  - handling annotations (mapping spans indexed on characters to token labels, transforming annotation schemes, etc.).
- [`demo_gene.ipynb`](.demo_gene.ipynb)
  - Demo of the main processes (loading corpus, content of the `File`s and `CorpusDict` methods, and generating `.tsv` files used for training ML models).

# Data structure

## Raw Corpus

In its raw version, the aggregated corpus is a `.json` file. For each file in the corpus it contains its raw text under `text`, as well as the DS spans (indexed on characters) stored in `labels` as a list of tuples `(start_id, end_id)`. Additional information, such as the name of the `original_corpus` and the `split` are also used to generate the `.tsv` versions.

## Custom data Classes

Two classes are introduced: `CorpusDict` which inherits from native python `dict` objects, and `File`.

**`CorpusDict`** are meant to be dictionnary with keys being file names and values `File` objects. They come mainly with two methods:
- `split_corpus` which return several `CorpusDict` objects, based on a given key (from `File` arguments) and values to be grouped. For instance it can be used to separate subcorpora or train/val/test splits.
- `merge_dfs_by_keys` to merge the `File`s `df_tokens` based on common keys (eg. data train/val/test splits)

**`File`** objects store the information of a single file. They are characterized by the file's name, and have the following **attributes**:
- `text` storing the text of the file
- `char_spans_labels` storing the annotations (spans of DS indexed on characters)
- `split` the train/val/test split associated with the file
- `original_corpus` the subcorpus from which the file originally comes from
- `df_tokens` storing tokenized text in a `DataFrame` together with the token idx (in terms of characters, where sentences start, as well as labels if available).

`File` have the **method**:
- `make_df_tokens` in order to turn the text into the `df_tokens` described above.


# Tokenizers

To make the framework able to deal with different text preprocessing libraries, `Document` and `Token` classes are defined so that tokenization methods can be made comparable to SpaCy's pipelines with a document containing a list of sentences, each containing a list of tokens characterized by both the word they represent and their position in the text (indexed in terms of characters).

Then, the NLTK regular expression tokenizers are adapted to fit these requirements as [`nltk_tokenizer_pipe`](#nltk_tokenizer_pipe). It offers an alternative to `spacy_nlp` that is also supported by the pipeline. Other sentence splitters and word-tokenizers could be included.

## `Document`

**Attributes:**
- `text`: contains raw text
- `sents`: contains list of sentences splitted from `text` (each sentence being a list of `Token`s)

## `Token`

**Attributes:**
- `idx`: index in terms of character (referring to `Token`'s position in a `Document` for instance)
- `text`: string of the word-token

## `nltk_tokenizer_pipe`

Implementation of a pipeline (sentence splitting + word tokenization) returning a `Document` from a raw text (`string`).

# Loading corpus

The method `load_corpus_from_json` will load the raw `json` corpus and turn it into a `CorpusDict` object storing the different entries into `File`s, including their tokenized text with annotations under the `df_tokens` field.

The text can either be tokenized using SpaCy pipeline by setting the method's following kwargs: `**{document_maker:spacy_nlp}`, or NLTK custom regex implementation: `**{document_maker:nltk_tokenizer_pipe}`.

# Generating Files

Train/Val/Test `.tsv` files, used to develop Machine Learning models can be computed and stored with the method `generate_tokens_df`. This will make files for eahc of the splits, also differenciating the "main" corpus from the OOD corpus comprising files from *Straight Talk!* subcorpus.

This can be done (with default arguments) by running the following command lines (first one for nltk tokenizer, second one spacy tokenizer):

```bash
python data_utils.py --data_dir 'DATA_DIR' --output_dir 'OUTPUT_DIR'
```

```bash
python data_utils.py --data_dir 'DATA_DIR' --output_dir 'OUTPUT_DIR' --tokenizer 'spacy_tokenization'
```
