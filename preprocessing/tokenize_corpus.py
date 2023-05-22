"""

*** data utils / data structure module ***

Content:

I. Data Structure — Classes Definition
    A. CorpusDict
        Object storing the corpus as a dictionnary
        with keys being file names and values File objects
    B. File
        Storing content of a file
        from its text and labels to is tokenized text in a DataFrame

II. Main methods to load and manipulate the raw corpus
    A. Loading raw json corpus into CorpusDict object
    B. Generating .tsv files per split used in ML algo
    
III. __main__
    Generate files for ML algo from command line

"""

import os

import argparse

import numpy as np
import pandas as pd
import random

from tqdm import tqdm

from itertools import groupby, count
import json

import re


# ========= SpaCy Tokenization

import spacy
spacy_nlp = spacy.load("fr_core_news_sm")


# ========= Custom methods

import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

# ========= NLTK Tokenization

from preprocessing.nltk_tokenizer import *

# ======== Conversions 

from preprocessing.utils_converters import *

# ========= DATA STRUCTURE

class CorpusDict(dict):
        
    def merge_dfs_by_keys(self
                          , key:str="split"
                          , values:list=["train", "val", "test"]
                          , add_file_name:bool = True
                          , save_files:bool = False
                          , output_dir:str = ""
                         )->dict:
        """
        Append all the <df_tokens> of all Files having the same <key> value.
        Rows marked with "EOF" separate the different files

        Arguments:
            key (str: "split")
                gathering field from the <File>s of the corpus
            values (list, default: ["train", "val", "test"])
                corresponding values of the <key>
            add_file_name (bool, default: True)
                add the "file" field in the merged_df_tokens containing the name of the file
            save_files (bool, default: False)
                wether or not to save the merged_df_tokens
            output_dir (str, default: "")
                [if <save_files> == True] path to the folder to save the `.tsv` merged dataframes
                
        Returns:
            (dict) {val1 : merged_df_tokens1, val2 : ...}
                     |                |
                segregating      merged <df_tokens>
                value from       of all files having 
                <values>         File.<key> == val
            
        Example:
            With the default values, a dictionnary of containing merged df_tokens for the train/val/test
            data splits will be returned: 
                {
                 "train": <df containing appended file.df_tokens from each file in train split>,
                 "val":   <df containing appended file.df_tokens from each file in val split>,
                 "test":  <df containing appended file.df_tokens from each file in test split>
                }
        """
        
        df_splits = {}

        for split_type in values:
            split_df = pd.DataFrame()
            file_map = []

            for file, file_name in zip(self.values(), self.keys()):
                if getattr(file, key) == split_type:
                    file_df = file.df_tokens

                    if add_file_name:
                        file_df["file"] = file.__name__

                    split_df = pd.concat([split_df, file_df])

                    # add "end of file" tag = "EOF"
                    split_df.loc[len(split_df)] = ["EOF"]*(len(split_df.columns))

            df_splits[split_type] = split_df
            
            if save_files:
                
                out_file_name = split_type+".tsv"
                
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                    
                print("Saving Dataframe with columns:\n {cols}\n composed of {nb_rows} rows at:\n{path}\n".format(
                    cols = list(split_df.columns),
                    path=output_dir+out_file_name,
                    nb_rows=len(split_df)

                )
                     )

                split_df.to_csv(path_or_buf=output_dir+out_file_name
                                    , sep="\t"
                                    , index=False
                                    #, header=False
                                   )

        return df_splits
    
    def split_corpus(self
                     , key:str="original_corpus"
                     , groups=[["fr-litbank", "synpaflex", "sini_bonus"], ["straight_talk"]]
                    )->tuple:
        
        """
        Splits the instance into several CorpusDict objects based on groups of ownership 
        defined for each File of the corpus (<key> could be "split", "original_corpus")

        Arguments:
            key (str: "original_corpus")
                segregating field from the <File>s of the corpus
            groups (list, default: [["fr-litbank", "synpaflex", "sini_bonus"], ["straight_talk"]])
                corresponding groups to segregate in different CorpusDict objects
                --> Will return CorpusDict object containing files for which their <key> are in the same element of <groups>
                
        Returns:
            tuple (CorpusDict, CorpusDict, ...)
                CorpusDict made of selected files from the current instance based on <key> and <groups>
            
        Example:
            With the default values, two CorpusDict object would be created from the current one:
                1. A corpus containing Files with <key> (original_corpus) 
                   contained in the 1st element of <groups> (ie. "fr-litbank", "synpaflex", "sini_bonus")
                2. A corpus containing Files with <key> (original_corpus) 
                   contained in the 2nd element of <groups> (ie. "straight_talk")
        """
        
        corpora = []
        for i, group in enumerate(groups):
            files_list = []
            
            for file in self.values():
                file_attr = getattr(file, key)
                if file_attr in group:
                    files_list += [file]
            
            corpora+=[CorpusDict({file.__name__:file for file in files_list})]
            
        return tuple(corpora)
        

class File:
    def __init__(self,
                 file_name:str,
                 raw_text:str,
                 split:str,
                 og_corpus:str="",
                 char_labels:list=[],
                 **kwargs 
                ):
        self.__name__ = file_name
        self.text = raw_text
        self.char_spans_labels = char_labels
        self.split = split
        self.original_corpus = og_corpus
        self.df_tokens = None
        
    def make_df_tokens(self
                       , document_maker = nltk_tokenizer_pipe
                       , annotated:bool = True
                       , label_name:str = "label"
                       , label_pos:str = "DS"
                       , label_neg:str = "O"
                       , annotation_scheme = None
                      )->None:
        """
        Inspired from function `convert_txtfile_to_dateframe` from Redewiedergabe

        The text of the File is word-tokenized and splitted into sentences.
        Outputs a pandas DataFrame with columns 'token', 'token_idx', 'sentstart', ; and 'label' if annotated.
        
        The resulting DataFrame is also stored in the field <df_tokens>

        Arguments:
            document_maker (default: nltk_tokenizer_pipe)
                tokenizer for the text
                - method/pipeline taking as input a raw <text> (string) 
                  and returning a Document object with field <sents> containing sentences
                  as a list of Token objects having fields <text> (string),
                  <idx> (index of the beginning of the Token in the text in terms of characters)
            annotated (bool, default: True)
                wether or not the File contains labels in field <char_spans_labels>
            label_name (str, default: "label")
                name given to the column of the output DataFrame containing the labels
            label_pos (str, default: "DS")
                positive labels (within the spans from field <char_spans_labels>)
            label_neg (str, default: "O")
                negative labels (outside of the spans from field <char_spans_labels>)
            annotation_scheme (str, default: None) --> /!\ supports only "BIO"
                annotation scheme to project labels from characters spans to word-tokens
                - None results in binary labels label_pos|label_neg
                
        Returns:
            None
            
            BUT the DataFrame is stored in field <df_tokens>:
            (pd.DataFrame): DataFrame containing the tokenized text and the projected labels
            Columns: 
                - "token": (str)                        --> textual content
                - "token_idx": (tuple: (int, int))      --> position of the token in text (start_id, end_id)
                                                            s.t. self.text[token_idx[0]:token_idx[1]] = token
                - "sentstart": (str) "yes"|"no"         --> "yes": beginning of a sentence | "no": not
                - "label": (str) [if <annotated>==True] --> label_pos in spans | label_neg otherwise
        """
        
        # 1. Make document (tokenize the text)
        doc = document_maker(self.text)
        
        # - Initialisation
        result = {"token": []
                  , "sentstart": []
                  , "token_idx": []
                 }
        
        # 2. Iterate over sentences
        for sent in doc.sents:
            # 2.1 Get tokens from the sentence and their indices (in terms of characters)
            sent_tokens = [tok.text for tok in sent]
            sent_indices = [tok.idx for tok in sent]
            token_indices = [(tok.idx, tok.idx+len(tok.text)) for tok in sent]
            
            # 2.2 Add first word of the sentence
            result["token"].append(sent_tokens[0])
            result["sentstart"].append("yes")
            result["token_idx"].append(token_indices[0])
            
            # 2.3 Add following tokens
            for tok, tok_id in zip(sent_tokens[1:], token_indices[1:]):
                result["token"].append(tok)
                result["token_idx"].append(tok_id)
                result["sentstart"].append("no")
                
        # 3. Turn the token/token_idx/sentstart lists into a pd.DataFrame
        result_df = pd.DataFrame(result)
        
        # 4. Retrieve the labels
        if annotated:
            # 4.1 Project the labels on spans of characters on the tokens
            df_labels = char_spans_to_tok_list_labels(df=result_df,
                                                      spans=self.char_spans_labels,
                                                      label_pos=label_pos,
                                                      label_neg=label_neg,
                                                      scheme = annotation_scheme
                                                     )
            # 4.2 Add a column in the DataFrame
            result_df[label_name] = df_labels
            
        # 5. Store the DataFrame in the File class field <df_tokens>
        self.df_tokens = result_df
        
        return
    
    
def load_corpus_from_json(json_path:str
                          , **kwargs
                         )->CorpusDict:
    """
    Loads raw corpus and returns a CorpusDict with each file tokenized.

    Arguments:
        json_path (str)
            path to the `json` corpus
            for each text, it must contain the fields:
                - "text" : (str)             --> Raw text
                - "labels": (list) [(tuple)] --> List of tuple: spans in terms of characters in text
                - "split": (str)             --> Text split ("train"/"val"/"test")
                - "original_corpus": (str)   --> Name of the original corpus
        **kwargs
            additional arguments --> given to <File.make_df_tokens> method
                - document_maker
                - annotated
                - label_name
                - label_pos
                - label_neg
                - annotation_scheme

    Returns:
        (CorpusDict): Corpus containing for each file the <text>, <char_spans_labels>, <split>, <original_corpus>
                      + <df_tokens> containing the tokenized text

    """
    # 1. Load raw corpus
    with open(json_path, 'r') as saved_corpus:
        agg_corpus = json.loads(saved_corpus.read())
    saved_corpus.close()
    
    # 2. Instantiate CorpusDict object to be returned
    corpus = CorpusDict()
    # 2.1 Iterate over each file from the loaded raw corpus
    for k, value in tqdm(zip(agg_corpus.keys(), agg_corpus.values())):
        # 2.2 Instantiate a File oject
        file = File(file_name=k,
                    raw_text=value["text"],
                    char_labels=value["labels"],
                    split=value["split"],
                    og_corpus = value["original_corpus"],
                   )
        # 2.3 Tokenize the file and make the <df_tokens>
        file.make_df_tokens(**kwargs)
        # 2.4 Store the expanded File in the CorpusDict
        corpus[k] = file
        
    return corpus


def generate_tokens_df(raw_data_path:str,
                       output_dir:str,
                       do_split:bool=True,
                       **kwargs
                      )->None:
    """
    Saves `.tsv` files for the train/val/test splits of both the "main" and "noisy" corpus.

    Arguments:
        raw_data_path (str: nltk_tokenizer_pipe) --> given to <load_corpus_from_json>
            path to the raw `json` corpus
            for each text, it must contain the fields:
                - "text" : (str)             --> Raw text
                - "labels": (list) [(tuple)] --> List of tuple: spans in terms of characters in text
                - "split": (str)             --> Text split ("train"/"val"/"test")
                - "original_corpus": (str)   --> Name of the original corpus
        output_dir (str)
            root of the folder where the `.tsv` files are saved 
        do_split (bool, default = True)
            Split the corpus into "main" / "noisy" based on the "original_corpus" key and predefined sets
        **kwargs
            additional arguments --> given to <File.make_df_tokens> method
                - document_maker
                - annotated
                - label_name
                - label_pos
                - label_neg
                - annotation_scheme

    Returns:
        None 
        
        BUT it will generate and save `.tsv` files for each of the data splits
            for both "main" and "noisy" corpus as predefined.
            The output folder will receive these files and have the following structure:
        
                <output_dir>/           
                ├── main_corpus/    --> fr-litbank, synpaflex, sini_bonus
                │   ├── test.tsv        --> test split (~0.8)
                │   ├── train.tsv       --> train split (~0.1)
                │   └── val.tsv         --> validation split (~0.1)
                └── noisy_corpus/     --> Straight Talk!
                   ├── test.tsv         --> Straight Talk! files
                   ├── train.tsv        --> EMPTY
                   └── val.tsv          --> EMPTY
                   
            The DataFrame have the following columns:
                - "token": (str)                        --> textual content
                - "token_idx": (tuple: (int, int))      --> position of the token in text (start_id, end_id)
                                                            s.t. self.text[token_idx[0]:token_idx[1]] = token
                - "sentstart": (str) "yes"|"no"         --> "yes": beginning of a sentence | "no": not
                - "label": (str) [if <annotated>==True] --> label_pos in spans | label_neg otherwise
                - "file": (str)                         --> name of the corresponding file
            
            + rows marked with "EOF" separate files from one another 
                

    """
    # 1. Load raw corpus + expand each file
    corpus = load_corpus_from_json(json_path=raw_data_path, **kwargs)
    print("Loaded corpus of {nb_files} files.".format(nb_files=len(corpus)))
    
    # 2. Split "main" / "noisy" sub-corpora
    if do_split:
        main_corpus, noisy_corpus = corpus.split_corpus()

        # 3. Merge DataFrames by split (train/val/test) 
        #    + Save them
        main_df_splits = main_corpus.merge_dfs_by_keys(save_files=True, output_dir=output_dir+"main_corpus/") 
        noisy_df_splits = noisy_corpus.merge_dfs_by_keys(save_files=True, output_dir=output_dir+"noisy_corpus/")
        
    # 2-bis. Keep corpus as loaded
    else:
        corpus.merge_dfs_by_keys(save_files=True, output_dir=output_dir) 
    
    return
    
    
    
if __name__ == '__main__':
    
    # 0. Parse arguments 
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', help='Path to the raw corpus in json format.')
    parser.add_argument('--output_dir', help='Path to the root folder of the generated .tsv files')
    parser.add_argument('--tokenizer', help='Document Maker (tokenizer)')
    parser.add_argument('--annotated', help='Are their spans in source corpus ?')
    parser.add_argument('--label_name', help='label column name')
    parser.add_argument('--label_pos', help='positive label name')
    parser.add_argument('--label_neg', help='negative label name')
    parser.add_argument('--annotation_scheme', help='how to project spans to discrete labels')
    parser.add_argument('--do_split', help='Split the aggregated corpus into "main"/"noisy"')
    
    
    args = parser.parse_args()
    
    if args.tokenizer == "spacy_tokenization":
        document_tokenizer_maker = spacy_nlp
    else: # elif ?
        document_tokenizer_maker = nltk_tokenizer_pipe
        if not args.tokenizer == "nltk_tokenization":
            args.tokenizer = "nltk_tokenization"
        
    if not args.label_pos:
        label_pos = "DS"
    else:
        label_pos = args.label_pos
        
    if not args.label_neg:
        label_neg = "O"
    else:
        label_neg = args.label_neg
        
    if not args.label_name:
        label_name = "label"
    else:
        label_name = args.label_name
        
    if not args.annotated:
        annotated = True
    else:
        annotated = args.annotated.lower() == 'true'
        
    if not args.do_split:
        do_split = True
    else:
        do_split = args.do_split.lower() == 'true'
        
    
    output_path = args.output_dir+args.tokenizer+"/"
    
    # 1. Parameters to make <df_tokens> for each of the File of the Corpus
    expander_args = {"document_maker":document_tokenizer_maker
                       , "annotated":annotated
                       , "label_name":label_name
                       , "label_pos":label_pos
                       , "label_neg":label_neg
                       , "annotation_scheme":args.annotation_scheme
                      }
    # 2. Save `.tsv` files for each split of both "main" and "ood" subcorpora
    generate_tokens_df(raw_data_path=args.data_dir
                       , output_dir=output_path
                       , do_split=do_split
                       ,**expander_args
                      )    
