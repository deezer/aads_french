""" 
*** Code from Redewiedergabe with minor adaptations ***
"""

from flair.models import SequenceTagger

import flair
from flair.data import Sentence
from flair.data import Corpus
#from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from flair.visual.training_curves import Plotter
from flair.embeddings import BertEmbeddings, WordEmbeddings, CharacterEmbeddings, StackedEmbeddings, FlairEmbeddings, TransformerWordEmbeddings, OneHotEmbeddings
from flair.data import Token

import sklearn.metrics
import os
import pandas as pd
import re
import datetime
import torch
import logging
import time

#import nltk.data
#from nltk.tokenize import word_tokenize
import numpy as np
import random

from tqdm import tqdm

#import spacy
#spacy.prefer_gpu()
#nlp = spacy.load("fr_core_news_sm")

from evaluation_schemes.evaluation import *
import json

from ml_utils import *
from results.results_utils import *

    
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%% Flair framework %%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def create_sentlist_from_file_batchmax(data:pd.DataFrame
                                       , maxlen=128
                                       , compare_column="label"
                                      ):
    
    """ ADAPTATION OF ml_utils.create_sentences_chunks to Flair framework
    
    ---> create_sentences_chunks
    Takes a pandas dataframe with columns 'token', 'token_idx' and 'sentstart' and creates a list of chunks.
    
    Each chunk may contain several real sentences, but at most <maxlen> tokens.
    Thus, the individual chunks are often shorter than <maxlen>.
    
    Sentences are appended to chunks in the reading directions (without optimization).
    Only sentences longer than <maxlen> can be splitted among several chunks.
    
    No file boundaries are crossed. Using "EOF" tags distinguishing rows associated with one file from another,
    each individual chunk can not contain excerpts from more than one file.
    <---
    The chunks are here turned into Flair objects: (Sentence) made of (Token)s
    

    Arguments:
        ---> given to create_sentences_chunks
        data (pd.DataFrame)
            tokenized text, DataFrame with required columns:
                - "token" (str)
                - "sentstart" (str: "yes"|"no")
                - "token_idx" (tuple: (int:tok_start_id, int:tok_end_id)
                - [if <compare_column> != "NaN"] => <compare_column>
        maxlen (int, default: 128)
            maximal length (in terms of word-tokens|rows) of the returned chunks
        compare_column (str, default: "label")
            Name of the column of <data> containing the labels
            - "NaN" for non labeled data
        <---
    Returns:
        (list) List of Flair's (Sentence)s complying with <create_sentences_chunks> conditions.
    """
    
    sent_list = []
    
    # relies on ml_utils
    chunks_list = create_sentences_chunks(data=data
                                          , maxlen=maxlen
                                          , compare_column=compare_column
                                          , eof_tag = "EOF"
                                         )
    # convert chunks to Sentence objects made of Token
    for chunk in chunks_list:
        toklist = chunk["tokens"]
        taglist = chunk["tags"]
        tokstarts = chunk["tokens_ids"]
        
        sent = Sentence(" ") # load with empy string to avoid Error and " " to avoid warning
        for i, tok in enumerate(toklist):
            flair_tok = Token(str(tok), start_position=tokstarts[i])
            flair_tok.set_label("label", str(taglist[i]))
            sent.add_token(flair_tok)
        if len(sent.tokens) > 0:
            sent_list.append(sent)
            
    return sent_list


def create_corpus(train_df:pd.DataFrame
                  , val_df:pd.DataFrame
                  , test_df:pd.DataFrame
                  , chunk_len:int=128
                 ):
    """
    Make a Flair (Corpus) object out of train/val/test dataframes
    
    Arguments:
        train_df, val_df, test_df (pd.DataFrame)
            tokenized text, DataFrame with required columns:
                - "token" (str)
                - "sentstart" (str: "yes"|"no")
                - "token_idx" (tuple: (int:tok_start_id, int:tok_end_id)
                - "label"
        ---> given to create_sentences_chunks
        chunk_len (int, default: 128)
            maximal length (in terms of word-tokens|rows) of the returned chunks
        <---
        
    Returns:
        (Corpus) Flair Corpus object 
    """

    train_list = create_sentlist_from_file_batchmax(train_df, maxlen=chunk_len)
    val_list = create_sentlist_from_file_batchmax(val_df, maxlen=chunk_len)
    test_list = create_sentlist_from_file_batchmax(test_df, maxlen=chunk_len)
    corpus: Corpus = Corpus(train_list, val_list, test_list)

    return corpus


def _get_embeddings(embtype="flair"
                    , corpus_for_dict=None
                   ):
    """
    *** This method is only needed when training your own models. It is not accessible from
    rwtagger_script and not documented in detail. Use at your own risk. ;-)
    If you want to use this method, you need to have access to the appropriate language
    embeddings and must adjust the paths accordingly ***
    get the embedding combination, specified by the value of embtype
    Specify new embedding types here 
    :param embtype: 
    :return: Tuple for embedding name and embedding (flair format)
    """
    if embtype == "mbert":
        # BERT embeddings
        _emb_name = "mBERT"
        _embeddings = BertEmbeddings('bert-base-multilingual-cased')

    elif embtype == "fasttext":
        # fasttext out of the box
        _emb_name = "Fasttext"
        _embeddings = WordEmbeddings('fr')
        
    elif embtype == "flair":
        # flair
        _emb_name = "Flair"
        _embeddings = StackedEmbeddings([FlairEmbeddings('fr-forward'),
                                         FlairEmbeddings('fr-backward'),
                                        ])
        
    elif embtype == "ohe":
        # One Hot Encoding (as a baseline for the architecture)
        _emb_name = "One-Hot Encoding"
        _embeddings = OneHotEmbeddings.from_corpus(corpus_for_dict)        
        
    elif embtype == "flair_stacked_fasttext":
            # Stacked Embeddings (Fasttext rwk_cbow_100 und Flair)
            _emb_name = "Stacked: Fasttext (fr), Flair (fr-forward), Flair (fr-backward)"
            _embeddings = StackedEmbeddings([
                WordEmbeddings("fr"),
                FlairEmbeddings('fr-forward'),
                FlairEmbeddings('fr-backward'),
            ])
        
    elif embtype == "flaubert": # ~~NOT WORKING !!~~
        # FlauBERT: https://huggingface.co/flaubert/flaubert_base_cased
        _emb_name = "FlauBERT"
        _embeddings = TransformerWordEmbeddings('flaubert/flaubert_base_cased'
                                                , allow_long_sentences=False # Otherwise not working
                                               )
        
    elif embtype == "camembert":
        # CamemBERT: https://huggingface.co/camembert-base
        _emb_name = "CamemBERT"
        _embeddings = TransformerWordEmbeddings('camembert-base'
                                                , allow_long_sentences=True
                                               )
        
    elif embtype == "barthez":
        # BARThez: https://huggingface.co/moussaKam/barthez
        _emb_name = "BARThez"
        _embeddings = TransformerWordEmbeddings('moussaKam/barthez')
        
    elif embtype == "cedille": # VERY HEAVY!
        # Cedille fr-boris: https://huggingface.co/Cedille/fr-boris
        _emb_name = "Cedille"
        _embeddings = TransformerWordEmbeddings('Cedille/fr-boris')

    else:
        print("Unknown embedding type: '{}'. Abort.".format(embtype))
        exit(1)
    return (_emb_name, _embeddings)


def rdwdgb_train(trainfile
                 , devfile
                 , testfile
                 , resfolder="./data/results"
                 , embtype="flair"
                 , use_crf=True
                 , rnn_layers = 2
                 , hidden_size = 256
                 , chunk_len=128
                 , lr=0.1
                 , batch_len=8
                 , max_epochs=8#150
                 , shuffle=False
                 , label_name="label"
                 , write_weights=True
                 , checkpoints=True
                 , plot=False
                ):
    """
    *** This method can be used to train new models with the settings used in project Redewiedergabe
    It is not accessible from rwtagger_script and not documented in detail. Use at your own risk. ;-)
    ***
    :param trainfile:
    :param devfile:
    :param testfile:
    :param resfolder:
    :param embtype:
    :param chunk_len:
    :param batch_len:
    :return:
    """
    if embtype == "ohe":
        emb_name, embeddings = _get_embeddings(embtype
                                               , corpus_for_dict=create_corpus(trainfile, devfile, testfile, chunk_len)
                                              )
    else:
        emb_name, embeddings = _get_embeddings(embtype
                                              )

    corpus: Corpus = create_corpus(trainfile, devfile, testfile, chunk_len)
    tag_dictionary = corpus.make_label_dictionary(label_type=label_name)

    if not os.path.exists(resfolder):
        os.makedirs(resfolder)

    tagger: SequenceTagger = SequenceTagger(hidden_size=hidden_size,
                                            embeddings=embeddings,
                                            tag_dictionary=tag_dictionary,
                                            tag_type=label_name,
                                            use_crf=use_crf,
                                            rnn_layers=rnn_layers
                                            )
    trainer: ModelTrainer = ModelTrainer(tagger, corpus)

    trainer.train(resfolder,
                  learning_rate=lr,
                  mini_batch_size=batch_len,
                  max_epochs=max_epochs,
                  shuffle=shuffle,
                  write_weights=write_weights,
                  checkpoint=checkpoints,
                  num_workers=0 # set for reproducibility
                 )
    # plot training curves
    if plot:
        plotter = Plotter()
        plotter.plot_training_curves(os.path.join(resfolder, 'loss.tsv'))
        plotter.plot_weights(os.path.join(resfolder, 'weights.txt'))
    
    
    
def rdwdgb_predict(data:pd.DataFrame,
                   model:SequenceTagger,
                   max_len:int,
                  )->pd.DataFrame:
    
    """ TODO : RE-COMMENT --> MAJOR MODIFICATION
    
    Adaptation from Redewiedergabe to Data Architecture.
    
    Given a dataframe <data> containing tokenized text,
    compute tokens' labels as predicted by the given model <model>.
    
    1. Chunk the text
    2. Run predictions on each chunk
    3. Append predicted labels to the given dataframe
    
    ---
    Arguments:
        data (pd.DataFrame)
            Tokenized text. Table must contain columns:
                - 'token' (str)
                - 'sentstart' (str: 'yes'|'no')
        model (SequenceTagger)
            Flair model: trained SequenceTagger
        max_len (int)
            Maximal length (in word-tokens) of the chunks given to the model.
            --> Should comply with the parameter set during <model> training.
        
    Returns:
        (pd.DataFrame) <data> augmented with a column named 'prediction'
                        containing the labels predicted by the <model>
    """
    sent_list = create_sentlist_from_file_batchmax(data,
                                                   maxlen=max_len,
                                                   compare_column="NaN"
                                                  )

    predictions = []
    for sent in tqdm(sent_list):
        model.predict(sent)
        pred_list = [tok.get_label().value for tok in sent]
        predictions += pred_list

    # Add the prediction to the given DataFrame
    sorted_eof_ind = sorted(data.index[data["token"]=="EOF"].to_list())
    preds_with_eof = predictions
    for eof_id in sorted_eof_ind:
        preds_with_eof = preds_with_eof[:eof_id]+["EOF"]+preds_with_eof[eof_id:]

    # Put the predictions in <data>
    data["prediction"] = preds_with_eof

    return data        
    
    
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%% Pipeline %%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
def run_flairModel(args
                   , train_corpus_name="train.tsv"
                   , val_corpus_name="val.tsv"
                   , test_corpus_name="test.tsv"
                  ):
    """
    Run Flair model based on given arguments
    """
        
    model_args, data_args, training_args, eval_args = args
        
    # Get arguments 
    output_folder = training_args.output_dir
    
    fixed_seed = training_args.seed
    lr = training_args.learning_rate
    embedding = model_args.embedding_name
    max_len = model_args.max_len
    batch_len = model_args.batch_len
    max_epochs = model_args.max_epochs
    do_shuffle = model_args.do_shuffle
    use_crf = model_args.use_crf
    rnn_layers = model_args.rnn_layers
    hidden_size = model_args.hidden_size
    
    label_column_name = data_args.label_column_name
    
    # Get corpus and splits  
    modelName = "flairModel_{embedding}".format(embedding=embedding
                                                 )
    
    path_save = output_folder+modelName
    
    # Fix the seed --> Seed needs to be fixed for all dependencies
    if fixed_seed!=None:
        logging.info("🌱 Setting Flair seed to {}".format(fixed_seed))
        flair.set_seed(fixed_seed)
        random.seed(fixed_seed)
        np.random.seed(fixed_seed)
        torch.manual_seed(fixed_seed)
        torch.backends.cudnn.enabled = False
        # if you are suing GPU
        torch.cuda.manual_seed(fixed_seed)
        torch.cuda.manual_seed_all(fixed_seed)
        
    if training_args.do_train:
        train_corpus = pd.read_csv(data_args.data_dir+"train.tsv", sep="\t")
    else:
        train_corpus = pd.DataFrame(columns=['token', 'sentstart', 'token_idx', 'label', 'file'])
        
    if training_args.do_eval:
        val_corpus = pd.read_csv(data_args.data_dir+"val.tsv", sep="\t")
    else:
        val_corpus = pd.DataFrame(columns=['token', 'sentstart', 'token_idx', 'label', 'file'])
        
    if training_args.do_predict:
        test_corpus = pd.read_csv(data_args.data_dir+"test.tsv", sep="\t")
    else:
        test_corpus = pd.DataFrame(columns=['token', 'sentstart', 'token_idx', 'label', 'file'])
    
    # Train model
    if training_args.do_train:
        logging.info("⏳ Training Flair Model")
        start_training_time = time.time()
        
        rdwdgb_train(trainfile=train_corpus
                     , devfile=val_corpus
                     , testfile=test_corpus
                     , resfolder=path_save
                     , embtype=embedding
                     , use_crf=use_crf
                     , rnn_layers = rnn_layers
                     , hidden_size = hidden_size
                     , chunk_len=max_len#100#128
                     , batch_len=batch_len#8
                     , max_epochs=max_epochs#3#10
                     , label_name=label_column_name#"label"
                     , shuffle=do_shuffle#False
                     , lr=lr
                    )
        
        training_time = time.time()-start_training_time
        training_time_str = time.strftime('%H:%M:%S', time.gmtime(training_time))
        logging.info(">>>> ⌛️ Training done in {}.".format(training_time_str))
    
    # PREDICTIONS --> on Validation AND/OR Test corpus
    # loading model if not given
    model_path = path_save+"/best-model.pt"
    
    if training_args.do_predict or training_args.do_eval:
        if not os.path.exists(model_path):
            logging.warning("Prediction aborted. Model not found at path '{}'. Please download a model and put it into "
                          "the appropriate directory. The model file must be named best-model.pt.".format(model_path))

        else:
            logging.info("loading model {}".format(model_path))
            model = SequenceTagger.load(model_path)
            logging.info("model loaded")
            
    if training_args.do_eval:
        # PREDICTIONS ON VALIDATION DATA FILE:
        logging.info("Predicting labels on VALIDATION split : ")
        start_predicting_time = time.time()
        
        val_corpus = rdwdgb_predict(data=val_corpus,
                                    model=model,
                                    max_len=max_len,
                                   )
        predicting_time = time.time()-start_predicting_time
        predicting_time_str = time.strftime('%H:%M:%S', time.gmtime(predicting_time))
        logging.info(">>>> ⏱ Predictions done in {}.".format(predicting_time_str))
        
        # Get overall GT/pred pair (without "EOF" tags)
        true_labels = list(val_corpus[val_corpus["token"]!="EOF"]["label"])
        predictions = list(val_corpus[val_corpus["token"]!="EOF"]["prediction"])
        
        # Break down per file predictions 
        per_file_dict = {}
        for flnm in np.unique(val_corpus[val_corpus["token"]!="EOF"]["file"]):
            file_df = val_corpus[val_corpus["file"]==flnm]
            
            per_file_dict[flnm] = {'gt_labels': list(file_df["label"])
                                   , 'pred_labels': list(file_df["prediction"])
                                  }
            
        # Run Evaluation
        logging.info("Evaluating model's performances on VALIDATION set :")
        start_evaluation_time = time.time()
        
        performances_evaluator = EvaluationChain(eval_args=eval_args,
                                                 ground_truth=true_labels,
                                                 predictions=predictions,
                                                 per_file_labels = per_file_dict,
                                                )
        results_dict = performances_evaluator.compute_scores()
        
        evaluation_time = time.time()-start_evaluation_time
        evaluation_time_str = time.strftime('%H:%M:%S', time.gmtime(evaluation_time))
        logging.info(">>>> ✅ Evaluation done in {}.".format(evaluation_time_str))
        
        # Display results
        logging.info(make_str_results_dict(results_dict, table_name = "~~ PERFORMANCES ON VALIDATION SET ~~"))
        logging.info("Storing evaluation results at: {res_path}".format(res_path=eval_args.eval_save_dir))
        # Save results
        performances_evaluator.save_results(prefix="val")
        # Store predictions
        if eval_args.save_prediction_files:
            val_corpus.to_csv(path_or_buf=eval_args.eval_save_dir+"val.tsv"
                              , sep="\t"
                              , index=False
                              #, header=False
                             )            
    
    if training_args.do_predict:
        # PREDICTIONS ON VALIDATION TEST FILE:
        logging.info("Predicting labels on TEST split : ")
        start_predicting_time = time.time()
        test_corpus = rdwdgb_predict(data=test_corpus,
                                     model=model,
                                     max_len=max_len,
                                    )
        predicting_time = time.time()-start_predicting_time
        predicting_time_str = time.strftime('%H:%M:%S', time.gmtime(predicting_time))
        logging.info(">>>> ⏱ Predictions done in {}.".format(predicting_time_str))
        
        # Get overall GT/pred pair (without "EOF" tags)
        true_labels = list(test_corpus[test_corpus["token"]!="EOF"]["label"])
        predictions = list(test_corpus[test_corpus["token"]!="EOF"]["prediction"])
        
        # Break down per file predictions 
        per_file_dict = {}
        for flnm in np.unique(test_corpus[test_corpus["token"]!="EOF"]["file"]):
            file_df = test_corpus[test_corpus["file"]==flnm]
            
            per_file_dict[flnm] = {'gt_labels': list(file_df["label"])
                                   , 'pred_labels': list(file_df["prediction"])
                                  }
            
        # Run Evaluation
        logging.info("Evaluating model's performances on TEST set :")
        start_evaluation_time = time.time()
        
        performances_evaluator = EvaluationChain(eval_args=eval_args,
                                                 ground_truth=true_labels,
                                                 predictions=predictions,
                                                 per_file_labels = per_file_dict,
                                                )
        results_dict = performances_evaluator.compute_scores()
        
        evaluation_time = time.time()-start_evaluation_time
        evaluation_time_str = time.strftime('%H:%M:%S', time.gmtime(evaluation_time))
        logging.info(">>>> ✅ Evaluation done in {}.".format(evaluation_time_str))
        
        # Display results
        logging.info(make_str_results_dict(results_dict, table_name = "~~ PERFORMANCES ON TEST SET ~~"))
        logging.info("Storing evaluation results at: {res_path}".format(res_path=eval_args.eval_save_dir))
        # Save results
        performances_evaluator.save_results(prefix="test")
        # Store predictions
        if eval_args.save_prediction_files:
            test_corpus.to_csv(path_or_buf=eval_args.eval_save_dir+"test.tsv"
                               , sep="\t"
                               , index=False
                               #, header=False
                              )
               
    return
    
    

