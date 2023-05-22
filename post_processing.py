import pandas as pd
import numpy as np

from evaluation_schemes.evaluation import *
from results.results_utils import *
from experiments_configs.configs_helpers import *
import argparse
import logging 

def is_punct(string:str)->bool:
    """
    True if string is not an alpha numeric character 
    (ie. a "word" from the text, but rather punctuation marking boundaries between clauses)
    """
    return string.isalnum() or "'" in string or "’" in string

def assign_clause_ids(df:pd.DataFrame)->pd.DataFrame:
    clause_ids = []

    curr_clause_id = 0
    prev_punct = False

    clause_ids += [curr_clause_id]
    for _, row in df[1:].iterrows():
        tok = row["token"]
        if tok=="EOF":
            clause_ids += [-1]
            curr_clause_id = 0
            prev_punct = False

        else:
            if not is_punct(tok):
                if "\n" in tok:
                    clause_ids += [-1]
                    prev_punct=False
                    curr_clause_id+=1
                else:
                    clause_ids += [curr_clause_id]
                    prev_punct=True

                curr_clause_id += 1
            else:
                #if prev_punct:
                 #   curr_clause_id += 1

                clause_ids += [curr_clause_id]
                prev_punct = False
                
    df["clause_id"] = clause_ids
    
    return df

def majority_label_per_clause(df:pd.DataFrame)->dict:
    file_clause_pred = {}

    for file in np.unique(df[df["file"]!="EOF"]["file"]):
        file_df = df[df["file"]==file]

        for c_id in np.unique(file_df[file_df["clause_id"]!=-1]["clause_id"]):
            # Get all preds for tokens in clause
            c_preds = file_df[file_df["clause_id"]==c_id]["prediction"]

            # Majority vote
            if len(np.unique(c_preds))>1:
                majority_lab = "DS" if (np.sum(c_preds=="DS")/len(c_preds))>=0.5 else "O"
            else:
                majority_lab = c_preds.iloc[0]

            # Store res
            if file in file_clause_pred.keys():
                file_clause_pred[file].update({c_id:majority_lab})
            else:
                file_clause_pred[file] = {c_id:majority_lab}
                
    return file_clause_pred

def post_process_cc_pred(path_to_pred_df:str
                        )->pd.DataFrame:
    
    loaded_df = pd.read_csv(path_to_pred_df, sep="\t")
    
    # 1. Assing an ID to all clauses of each file in the loaded table
    loaded_df = assign_clause_ids(loaded_df)
    
    # 2. Assign 1 unique label per clause (through majority vote)
    file_clause_pred = majority_label_per_clause(loaded_df)
    
    # 3. Add Clause-Continuous prediction to the loaded table

    clause_preds = []

    for _, row in loaded_df.iterrows():
        if row["token"]=="EOF":
            clause_preds += ["EOF"]
        else:
            c_id = row["clause_id"]
            if c_id == -1:
                clause_preds += [row["prediction"]]
            else:
                clause_preds += [file_clause_pred[row["file"]][c_id]]

    loaded_df["cc_prediction"] = clause_preds
    
    return loaded_df


def majority_vote(best_per_files) : 
    majority = []
    for metric in best_per_files : 
        if len([i for i in metric if i == "TRAD"]) > len([i for i in metric if i != "TRAD"]) : 
            majority.append("TRAD")
        elif len([i for i in metric if i == "TRAD"]) == len([i for i in metric if i != "TRAD"]) : 
            majority.append("EQUAL") 
        else : 
            majority.append("CC")
    return majority 

def postprocessing_results(loaded_df
                      , file_eval_args
                     )->None:

    unique_files = np.unique(loaded_df[loaded_df["file"]!="EOF"]["file"])
    results= np.empty((13,5), dtype="object")
    
    # Averaged over files
    results_averaged = [[] for _ in range(13)]
    best_averaged = [[] for _ in range(13)]
    for file in unique_files:
        file_df = loaded_df[loaded_df["file"]==file]
        
        file_preds_perf = EvaluationChain(eval_args=file_eval_args,
                                          ground_truth=list(file_df["label"]),
                                          predictions=list(file_df["prediction"]),
                                         )
        preds_perf = file_preds_perf.compute_scores()
        
        file_cc_preds_perf = EvaluationChain(eval_args=file_eval_args,
                                          ground_truth=list(file_df["label"]),
                                          predictions=list(file_df["cc_prediction"]),
                                         )
        cc_preds_perf = file_cc_preds_perf.compute_scores()

        for idx, (s_trad, s_cc) in enumerate(zip(preds_perf.values(), cc_preds_perf.values())):
            results_averaged[idx].append(s_cc)
            if idx == len(preds_perf) - 1:
                best_val="TRAD" if s_trad<s_cc else "CC"
            else:
                best_val="TRAD" if s_trad>s_cc else "CC"
            best_averaged[idx].append(best_val)

    results_averaged = np.asarray(results_averaged)
    results_averaged_mean = np.mean(results_averaged, axis = 1)
    results_averaged_std = np.std(results_averaged, axis = 1)
    
    results[:,2] = results_averaged_mean
    results[:,3] = results_averaged_std
    results[:,4] = majority_vote(best_averaged)
    # Overall:
    preds_perf = EvaluationChain(eval_args=file_eval_args,
                ground_truth=list(loaded_df[loaded_df["token"]!="EOF"]["label"]),
                predictions=list(loaded_df[loaded_df["token"]!="EOF"]["prediction"]),
               ).compute_scores()
    
    cc_preds_perf = EvaluationChain(eval_args=file_eval_args,
                    ground_truth=list(loaded_df[loaded_df["token"]!="EOF"]["label"]),
                    predictions=list(loaded_df[loaded_df["token"]!="EOF"]["cc_prediction"]),
                   ).compute_scores()

    results[:,0] = np.asarray(list(cc_preds_perf.values()))
    
    for idx, (s_trad, s_cc) in enumerate(zip(preds_perf.values(), cc_preds_perf.values())):

            if idx == len(preds_perf) - 1:
                best_val="TRAD" if s_trad<s_cc else "CC"
            else:
                best_val="TRAD" if s_trad>s_cc else "CC"
            results[idx, 1] = best_val

    results_df = pd.DataFrame(results, index = cc_preds_perf.keys(), columns = ["Overall", "Overall-best", "Averaged-mean", "Averaged-std", "Averaged-best"])
    return results_df
    


if __name__ == "__main__" :     
    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--result_path'
                        , help='Folder where predictions tsv files are stored'
                        , default="results/best/"
                       )
   
    args = parser.parse_args()
    
    available_test_corpus = os.listdir(args.result_path)
    
    logging.info("Clause-Continous Post-Processing predictions:")
    
    for dir in available_test_corpus : 
        logging.info(f"\t{os.path.join(args.result_path,dir)}")
        length = len(os.path.join(args.result_path,dir)) +2
        for model in os.listdir(os.path.join(args.result_path,dir)) : 
            logging.info((" "*length)+ f"\t{model}")
            
    # Post Process Test Results
    
    file_eval_args=EvalArguments(token_level_precision_recall_fscore=True,
                             strict_match_precision_recall_fscore=True,
                             purity_coverage=True,
                             FairEval_precision_recall_fscore=True,
                             zonemap=True
                            )
    
    for test_corpus in available_test_corpus : 
        models = os.listdir(os.path.join(args.result_path,dir))
        for model in models : 
            logging.info(f"===========================\n", "Processing {os.path.join(args.result_path, test_corpus, model, 'test.tsv')}")
            result_cc = post_process_cc_pred(os.path.join(args.result_path, test_corpus, model, "test.tsv"))
            results_df = postprocessing_results(result_cc, file_eval_args)
            results_df.to_csv(os.path.join(args.result_path, test_corpus, model, "test_cc.tsv"), sep = "\t", index=True, header=True)