"""
    *** Handling new content ***
- .json corpus from .txt files
- 
"""

import os
from os import path

import json
import pandas as pd


import logging
logging.getLogger().setLevel(logging.INFO)



def make_json_from_texts(folder_path:str
                         , output_dir:str=""
                         , new_corpus_name:str="new_corpus"
                        )->None:
    logging.info("📚 Turning .txt files in {f} to .json corpus.".format(f=folder_path))
    
    files_list = []

    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            files_list += [file]
            
    logging.info("\t{nb} .txt files found in folder:\n\t\t\t{str_list}".format(nb=len(files_list),
                                                                               str_list="\n\t\t\t".join(files_list)
                                                                              ))

    txts_list = []
    for txt_file in files_list:
        with open(path.join(folder_path, txt_file), 'r') as file:
            txts_list += [file.read()]
        file.close()

    corpus_dict = {}

    for txt, name in zip(txts_list, files_list):
        name = name.replace(".txt", "")
        corpus_dict[name] = {}
        corpus_dict[name]["text"] = txt
        corpus_dict[name]["labels"] = []
        corpus_dict[name]["original_corpus"] = "new_content"
        corpus_dict[name]["split"] = "test"

    with open(os.path.join(output_dir, new_corpus_name+".json"), 'w+') as fp:
        json.dump(corpus_dict, fp)
    fp.close()
    
    logging.info("📨 New .json corpus stored at {path}".format(path=os.path.join(output_dir, new_corpus_name+".json")))
                
    return 


def merge_predictions(methods:list = ["regex", "flair", "transformer"],
                      save_merged_df_path:str="output/merged_predictions.tsv"
                     )->pd.DataFrame:
    # 1. Read saved files for each method
    methods_dfs = []
    for method in methods: # ! saved prediction files must comply with the strict format used to read them below
        methods_dfs += [pd.read_csv("output/"+method+"/test.tsv", sep="\t")]
        
    # 2. Put the predictions from each model in the same DataFrame
    merged_df = methods_dfs[0][["file", "token", "sentstart", "token_idx"]].copy()
    for df, method in zip(methods_dfs, methods):
        merged_df["pred_"+method] = df["prediction"]
        
    # 3. Save the merged DataFrame if needed
    if save_merged_df_path:
        merged_df.to_csv(save_merged_df_path, sep="\t")
        logging.info("Tokenized text with merged predictions saved at {}".format(save_merged_df_path))
        
    return merged_df