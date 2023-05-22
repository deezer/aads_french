from .rb_quote import *
from evaluation_schemes.evaluation import *
from preprocessing.data_utils import *
from results.results_utils import *
from tqdm import tqdm
import time
import copy

import logging

def run_regexModel(args,
                   #corpus_to_test
                  ):
    logging.getLogger().setLevel(logging.INFO)
    
    model_args, data_args, training_args, eval_args = args
    
    # 1. Loading data

    PATH_DATA = data_args.data_dir
    
    logging.info("Loading corpus from {path} with {tokenizer_name} ...".format(
        path=PATH_DATA
        , tokenizer_name=model_args.word_tokenizer
    ))
    
    if model_args.word_tokenizer=="spacy_tokenization":
        document_maker = spacy_nlp
    else:
        document_maker = nltk_tokenizer_pipe
    
    # 1.1 Tokenize the texts
    #     + expand each file of the corpus (make <df_tokens>)
    logging.info("⏳ Preprocessing the corpus.")
    start_preprocessing_time = time.time()
    corpus = load_corpus_from_json(PATH_DATA, **{'document_maker':document_maker})
    # 1.3 split Main / Noisy (ST!) based on original corpus names
    main_corpus, noisy_corpus = corpus.split_corpus()
    # 1.4 split Train/Val/Test 
    if eval_args.eval_on_main: 
        train_corpus, val_corpus, test_corpus = main_corpus.split_corpus(key="split"
                                                                         , groups = [["train"], ["val"], ["test"]]
                                                                        )
    elif eval_args.eval_on_noisy:
        train_corpus, val_corpus, test_corpus = noisy_corpus.split_corpus(key="split"
                                                                        , groups = [["train"], ["val"], ["test"]]
                                                                       )
    else: # Consider only a test set
        train_corpus, val_corpus, test_corpus = CorpusDict(), CorpusDict(), copy.deepcopy(corpus)
        
    preprocessing_time = time.time()-start_preprocessing_time
    preprocessing_time_str = time.strftime('%H:%M:%S', time.gmtime(preprocessing_time))
    logging.info(">>>> ⌛️ Preprocessing done in {}.".format(preprocessing_time_str))
    
    # 2. Building RegEx Model
    regexType = model_args.regex_type
    
    if regexType == "bookNLP":
        if model_args.dash_threshold is not None:
            dash_threshold = model_args.dash_threshold
        else:
            dash_threshold = 0.
            
        regex_method = lambda x: bookNLP_like_quotes(corpus_file=x,
                                                     prop_th=dash_threshold
                                                    )
    elif regexType == "regexByszuk":
        regex_method = byszuk_baseline
        
        
    # 3. Run RegEx (W)DSR on set of texts
    # 3.1 Evaluation set
    if training_args.do_eval and len(val_corpus)>0:
        
        logging.info("Running Regular Expression on Validation set:")
        start_prediction_time = time.time()
        
        per_file_dict = {}
        
        for k in tqdm(val_corpus.keys()):
            
            file_df = val_corpus[k].df_tokens

            pred_spans = regex_method(val_corpus[k])
            
            predicted_labels = char_spans_to_tok_list_labels(df = file_df
                                                             , spans = pred_spans
                                                            )
            
            per_file_dict[k] = {'gt_labels': list(file_df["label"])
                                , 'pred_labels': predicted_labels
                               }
            # Store the performances in <File.df_tokens>
            val_corpus[k].df_tokens["prediction"] = predicted_labels
            
        prediction_time = time.time()-start_prediction_time
        prediction_time_str = time.strftime('%H:%M:%S', time.gmtime(prediction_time))
        logging.info(">>>> 💨 Predictions done in {}.".format(prediction_time_str))
                
        # 4. Evaluate the performances
        logging.info("Evaluating model's performances on VALIDATION set :")
        start_evaluation_time = time.time()
        performances_evaluator = EvaluationChain(eval_args=eval_args,
                                                 per_file_labels=per_file_dict
                                                )
        results_dict = performances_evaluator.compute_scores()
        evaluation_time = time.time()-start_evaluation_time
        evaluation_time_str = time.strftime('%H:%M:%S', time.gmtime(evaluation_time))
        logging.info(">>>> ✅ Evaluation done in {}.".format(evaluation_time_str))
        
        logging.info(make_str_results_dict(results_dict, table_name = "~~ PERFORMANCES ON VALIDATION SET ~~"))
        logging.info("Storing evaluation results at: {res_path}".format(res_path=eval_args.eval_save_dir))
        # 4.1 Save scores
        performances_evaluator.save_results(prefix="val")
        # 4.2 Save predictions
        if eval_args.save_prediction_files:
            val_corpus.merge_dfs_by_keys(key="split"
                                         , values=["val"]
                                         , add_file_name = True
                                         , save_files = True
                                         , output_dir = eval_args.eval_save_dir
                                        )
        
        
    # 3.2 Test set   
    if training_args.do_predict and len(test_corpus)>0:
        
        logging.info("Running Regular Expression on Test set:".format(path=PATH_DATA))
        start_prediction_time = time.time()
        
        per_file_dict = {}
        
        for k in tqdm(test_corpus.keys()):

            file_df = test_corpus[k].df_tokens

            pred_spans = regex_method(test_corpus[k])
            
            predicted_labels = char_spans_to_tok_list_labels(df = file_df
                                                             , spans = pred_spans
                                                            )
            
            per_file_dict[k] = {'gt_labels': list(file_df["label"])
                                , 'pred_labels': predicted_labels
                               }
            # Store the performances in <File.df_tokens>
            test_corpus[k].df_tokens["prediction"] = predicted_labels
            
        prediction_time = time.time()-start_prediction_time
        prediction_time_str = time.strftime('%H:%M:%S', time.gmtime(prediction_time))
        logging.info(">>>> 💨 Predictions done in {}.".format(prediction_time_str))
                
        # 4. Evaluate the performances
        logging.info("Evaluating model's performances on TEST set :")
        start_evaluation_time = time.time()
        performances_evaluator = EvaluationChain(eval_args=eval_args,
                                                 per_file_labels=per_file_dict
                                                )
        results_dict = performances_evaluator.compute_scores()
        evaluation_time = time.time()-start_evaluation_time
        evaluation_time_str = time.strftime('%H:%M:%S', time.gmtime(evaluation_time))
        logging.info(">>>> ✅ Evaluation done in {}.".format(evaluation_time_str))
        
        logging.info(make_str_results_dict(results_dict, table_name = "~~ PERFORMANCES ON TEST SET ~~"))
        logging.info("Storing test results at: {res_path}".format(res_path=eval_args.eval_save_dir))
        # 4.1 Save scores
        performances_evaluator.save_results(prefix="test")
        # 4.2 Save predictions
        if eval_args.save_prediction_files:
            test_corpus.merge_dfs_by_keys(key="split"
                                          , values=["test"]
                                          , add_file_name = True
                                          , save_files = True
                                          , output_dir = eval_args.eval_save_dir
                                         )
        
    return 