from datasets import Dataset
from datasets import ClassLabel, load_dataset, load_metric

import os
import time

import datasets
import numpy as np
import torch.nn as nn
from datasets import ClassLabel, load_dataset, load_metric

import transformers
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    HfArgumentParser,
    PretrainedConfig,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from copy import deepcopy
#from tabulate import tabulate

#from sklearn.metrics import precision_recall_fscore_support
#from datasets import load_metric

import sys
#sys.path.append('../')
from evaluation_schemes.evaluation import *

from ml_utils import *
from results.results_utils import *

_SPECIAL_TOKEN = "[UK]"

def run_transformerModel(args
                         , corpus_to_test={}
                        ):
    
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # %%%%%%%%% HELPER FUNCTIONS %%%%%%%%%%%%%
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def words_from_subtokens(subtokens:list
                            )->list:
        """
        Find back word-tokens from subtokens list (after tokenizer.tokenize([word-tokens list], is_split_into_words=True) 
        """
        word_toks_from_tokenizer = []
        word = ""
        for subtok in subtokens:
            if subtok.startswith("▁"):
                word_toks_from_tokenizer += [word]
                word = subtok.replace("▁", "")
            else:
                word+=subtok

        if word:
            word_toks_from_tokenizer += [word]
        word_toks_from_tokenizer = word_toks_from_tokenizer[1:]

        return word_toks_from_tokenizer
    
    def clean_tokens_list(tokens_list:list,
                         special_token:str=_SPECIAL_TOKEN
                        )->list:
        """
        Replace word-tokens not recognized by the tokenizer with _SPECIAL_TOKEN
        """
        new_tokens = [tok 
                      if tok in words_from_subtokens(tokenizer.tokenize([tok], is_split_into_words=True))
                      else special_token
                      for tok in tokens_list
                     ]

        return new_tokens
    
    def clean_examples(examples
                      ):
        """
        From a batch, clean its examples, meaning:
            - replace word-tokens unknown by the tokenizer (eg. "\n", "\xa0", ...) with a special token
            - split examples for which sub-tokenization (by the <tokenizer>) results in too long sequences for the model
            
        => Results in a new list of examples with replaced unknown tokens and splitted examples if needed
            ( 2*len( original examples ) >= len( new examples ) >= len( original examples ) )
              |
              └── TODO: recursive implementation to handle any length of the original input sequences
        """
        curr_id = 0

        new_tokens_chunks=[]
        new_tags_chunks=[]
        new_ids = []

        for ex_id, ex_toks, ex_tags in zip(examples["id"], examples[text_column_name], examples[label_column_name]): #TODO : args
            # REMOVE UNKNOWN WORD-TOKENS
            cleaned_tokens = clean_tokens_list(ex_toks)
            # SUB-TOKENIZE TEXT
            subtokens = tokenizer.tokenize(cleaned_tokens, is_split_into_words=True)

            # ! will not work if very long input with len(subtokens)>2*data_args.max_seq_length (would require recursive loop)
            model_max_subtokens_length = data_args.max_seq_length
            if len(subtokens)>model_max_subtokens_length-2: #TODO : args
                # map subtokens with word-tokens
                word_token_ids = np.cumsum([subtok.startswith("▁") for subtok in subtokens])-1
                # cut word-tokens list s.t. the subtokens list is cut in half
                cut_toks_at = word_token_ids[int(len(subtokens)/2)]

                # Make two examples out of the original one
                new_tokens_chunks += [cleaned_tokens[:cut_toks_at], cleaned_tokens[cut_toks_at:]]
                new_tags_chunks += [ex_tags[:cut_toks_at], ex_tags[cut_toks_at:]]
                new_ids += [curr_id, curr_id+1]
                curr_id += 2

            else:
                # no problem: keep example as is
                new_tokens_chunks += [cleaned_tokens]
                new_tags_chunks += [ex_tags]
                new_ids += [curr_id]
                curr_id += 1

        # store new cleaned and shortened examples
        examples["id"]=new_ids
        examples[text_column_name]=new_tokens_chunks
        examples[label_column_name]=new_tags_chunks

        return examples
    
    
    def tokenize_and_align_labels(examples):
        """ Tokenize all texts and align the labels with them. """
        tokenized_inputs = tokenizer(
            examples[text_column_name],
            padding=padding,
            truncation=True,
            max_length=data_args.max_seq_length,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
        )
        labels = []
        for i, label in enumerate(examples[label_column_name]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label_to_id[label[word_idx]])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    if data_args.label_all_tokens:
                        label_ids.append(b_to_i_label[label_to_id[label[word_idx]]])
                    else:
                        label_ids.append(-100)
                previous_word_idx = word_idx

            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    """ In the event the labels are not a `Sequence[ClassLabel]`, we will need to go through the dataset to get the unique labels. """
    def get_label_list(labels):
        unique_labels = set()
        for label in labels:
            unique_labels = unique_labels | set(label)
        label_list = list(unique_labels)
        label_list.sort()
        return label_list
    
    """ get configs arguments """
    model_args, data_args, training_args, eval_args = args
    
    # instantiate compute_metrics method w/ parsed evaluation arguments:
    def compute_metrics(p
                        , eval_args=eval_args
                       )->dict:
        
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        
        flat_pred = [
            label_list[p]
            for prediction, label in zip(predictions, labels)
            for (p, l) in zip(prediction, label) if l != -100
        ]

        flat_gt = [
            label_list[l]
            for prediction, label in zip(predictions, labels)
            for (p, l) in zip(prediction, label) if l != -100
        ]
        
        performances_evaluator = EvaluationChain(ground_truth=flat_gt,
                                             predictions=flat_pred,
                                             eval_args=eval_args
                                            )

        return performances_evaluator.compute_scores()
    
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # %%%%%%%%%%% INITIALISATION %%%%%%%%%%%%%
    # %%%%%%% DATA + MODEL & TOKENIZER %%%%%%%
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    logging.info("⏳ Loading and preparing Dataset")
    start_preprocessing_time = time.time()
    
    # load data
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir="~/.cache/huggingface/datasets/{}".format(model_args.max_chunk_len),#model_args.cache_dir,
            data_dir=data_args.data_dir,
            **{'max_chunk_len':model_args.max_chunk_len}
        )
         
    # Set seed before initializing model.
    set_seed(training_args.seed)
    
    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logging.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
            
    
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.
    column_names = raw_datasets["test"].column_names
    features = raw_datasets["test"].features

    
    if data_args.text_column_name is not None:
        text_column_name = data_args.text_column_name
    elif "tokens" in column_names:
        text_column_name = "tokens"
    else:
        text_column_name = column_names[0]
        
    
    if data_args.label_column_name is not None:
        label_column_name = data_args.label_column_name
    elif f"{data_args.task_name}_tags" in column_names:
        label_column_name = f"{data_args.task_name}_tags"
    else:
        label_column_name = column_names[1]
    
    
    # If the labels are of type ClassLabel, they are already integers and we have the map stored somewhere.
    # Otherwise, we have to get the list of labels manually.
    labels_are_int = isinstance(features[label_column_name].feature, ClassLabel)
    if labels_are_int:
        label_list = features[label_column_name].feature.names
        label_to_id = {i: i for i in range(len(label_list))}
    else:
        label_list = get_label_list(raw_datasets["train"][label_column_name])
        label_to_id = {l: i for i, l in enumerate(label_list)}

    num_labels = len(label_list)
    
    
    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    
    # If <do_train> --> set model ready for training
    # Else --> load model from <training_args.output_dir>
    if training_args.do_train:
        config = AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            num_labels=num_labels,
            finetuning_task=data_args.task_name,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

        tokenizer_name_or_path = model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path
        if config.model_type in {"gpt2", "roberta"}:
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name_or_path,
                cache_dir=model_args.cache_dir,
                use_fast=True,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                add_prefix_space=True,
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name_or_path,
                cache_dir=model_args.cache_dir,
                use_fast=True,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
            )

        model = AutoModelForTokenClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

        # Tokenizer check: this script requires a fast tokenizer.
        if not isinstance(tokenizer, PreTrainedTokenizerFast):
            raise ValueError(
                "This example script only works for models that have a fast tokenizer. Checkout the big table of models "
                "at https://huggingface.co/transformers/index.html#supported-frameworks to find the model types that meet this "
                "requirement"
            )
    else:
        model = AutoModelForTokenClassification.from_pretrained(training_args.output_dir, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(training_args.output_dir, local_files_only=True)
        
    """
    Added segment to handle:
        - unrecognized word-tokens
        - subtokens sequences longer than model.max_len
    """
    for k in raw_datasets.keys():
        raw_datasets[k] = raw_datasets[k].map(clean_examples
                                              , batched=True
                                              , desc="🧹 Cleaning examples in {split} dataset".format(split=k)
                                             )
    
    """
    Back to usual code
    """
        
    
    # Model has labels -> use them.
    if model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id:
        if list(sorted(model.config.label2id.keys())) == list(sorted(label_list)):
            # Reorganize `label_list` to match the ordering of the model.
            if labels_are_int:
                label_to_id = {i: int(model.config.label2id[l]) for i, l in enumerate(label_list)}
                label_list = [model.config.id2label[i] for i in range(num_labels)]
            else:
                label_list = [model.config.id2label[i] for i in range(num_labels)]
                label_to_id = {l: i for i, l in enumerate(label_list)}
        else:
            logging.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(model.config.label2id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
            
    # Set the correspondences label/ID inside the model config
    model.config.label2id = {l: i for i, l in enumerate(label_list)}
    model.config.id2label = {i: l for i, l in enumerate(label_list)}
    
    # Map that sends B-Xxx label to its I-Xxx counterpart
    b_to_i_label = []
    for idx, label in enumerate(label_list):
        if label.startswith("B-") and label.replace("B-", "I-") in label_list:
            b_to_i_label.append(label_list.index(label.replace("B-", "I-")))
        else:
            b_to_i_label.append(idx)
            
    # Preprocessing the dataset
    # Padding strategy
    padding = "max_length" if data_args.pad_to_max_length else False
    
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                tokenize_and_align_labels,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )
            
    if training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                tokenize_and_align_labels,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )
            
    if training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                tokenize_and_align_labels,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )
            
    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None)
    
    
    if data_args.reinit_layers > 0:
        # if Pooler layer exists reinit it too
        encoder_temp = getattr(model
                               , "roberta"#config.model_type#"roberta" #config.model_type
                              )
        if encoder_temp.pooler is not None:
            encoder_temp.pooler.dense.weight.data.normal_(mean=0.0, std=encoder_temp.config.initializer_range)
            encoder_temp.pooler.dense.bias.data.zero_()
            for p in encoder_temp.pooler.parameters():
                p.requires_grad = True

        BertLayerNorm = nn.LayerNorm
        reinit_layers = data_args.reinit_layers
        for layer in encoder_temp.encoder.layer[-reinit_layers :]:
            for module in layer.modules():
                if isinstance(module, (nn.Linear, nn.Embedding)):
                    # Slightly different from the TF version which uses truncated_normal for initialization
                    # cf https://github.com/pytorch/pytorch/pull/5617
                    module.weight.data.normal_(mean=0.0, std=encoder_temp.config.initializer_range)
                elif isinstance(module, BertLayerNorm):
                    module.bias.data.zero_()
                    module.weight.data.fill_(1.0)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()
      
    preprocessing_time = time.time()-start_preprocessing_time
    preprocessing_time_str = time.strftime('%H:%M:%S', time.gmtime(preprocessing_time))
    logging.info(">>>> ⌛️ Preprocessing done in {}.".format(preprocessing_time_str))
                    
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # %%%%%%%%%%%%%%%  MODEL  %%%%%%%%%%%%%%%%
    # %%%%%%%%%  TRAIN + VAL + PREDS  %%%%%%%%
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Training
    if training_args.do_train:
        logging.info("🤗 Training HuggingFace Model")
        start_training_time = time.time()
        
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
            
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        trainer.save_model()  # Saves the tokenizer too for easy upload
        
        training_time = time.time()-start_training_time
        training_time_str = time.strftime('%H:%M:%S', time.gmtime(training_time))
        logging.info(">>>> 🤗 Training done in {}.".format(training_time_str))

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        
    # Evaluation
    if training_args.do_eval: # MODIFIED TO FIT FRAMEWORK
        logging.info("*** Evaluate ***")
        split_name = "val"
        
        start_predicting_time = time.time()
        
        predictions, pred_labels, pred_metrics = trainer.predict(eval_dataset, metric_key_prefix=split_name)
        predictions = np.argmax(predictions, axis=2)
        
        predicting_time = time.time()-start_predicting_time
        predicting_time_str = time.strftime('%H:%M:%S', time.gmtime(predicting_time))
        logging.info(">>>> ⏱ Predictions done in {}.".format(predicting_time_str))
        
        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, pred_labels)
            ]
        flattened_preds = [p for preds in true_predictions for p in preds]
        
        # map predictions on loaded DataFrame
        loaded_df = pd.read_csv(data_args.data_dir+split_name+".tsv", sep="\t")

        sorted_eof_ind = sorted(loaded_df.index[loaded_df["token"]=="EOF"].to_list())
        preds_with_eof = flattened_preds
        for eof_id in sorted_eof_ind:
            preds_with_eof = preds_with_eof[:eof_id]+["EOF"]+preds_with_eof[eof_id:]

        # Put the predictions in <data>
        loaded_df["prediction"] = preds_with_eof
        
        # Break down per file predictions 
        per_file_dict = {}
        for flnm in np.unique(loaded_df[loaded_df["token"]!="EOF"]["file"]):
            file_df = loaded_df[loaded_df["file"]==flnm]

            per_file_dict[flnm] = {'gt_labels': list(file_df["label"])
                                   , 'pred_labels': list(file_df["prediction"])
                                  }
        
        logging.info("📊 Computing performances")
        start_evaluation_time = time.time()
        # Run Evaluation
        performances_evaluator = EvaluationChain(eval_args=eval_args,
                                                 ground_truth=list(loaded_df[loaded_df["token"]!="EOF"]["label"]),
                                                 predictions=list(loaded_df[loaded_df["token"]!="EOF"]["prediction"]),
                                                 per_file_labels = per_file_dict,
                                                )
        results_dict = performances_evaluator.compute_scores()
        
        evaluation_time = time.time()-start_evaluation_time
        evaluation_time_str = time.strftime('%H:%M:%S', time.gmtime(evaluation_time))
        logging.info(">>>> ✅ Evaluation done in {}.".format(evaluation_time_str))
        
        # Display results
        logging.info(make_str_results_dict(results_dict, table_name = "~~ PERFORMANCES ON "+split_name.upper()+" SET ~~"))
        logging.info("Storing evaluation results at: {res_path}".format(res_path=eval_args.eval_save_dir))
        # Save results
        performances_evaluator.save_results(prefix=split_name)
        # Store predictions
        if eval_args.save_prediction_files:
            loaded_df.to_csv(path_or_buf=eval_args.eval_save_dir+split_name+".tsv"
                             , sep="\t"
                             , index=False
                             #, header=False
                            )
        

        #metrics = trainer.evaluate()

        #max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        #metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        #trainer.log_metrics("eval", metrics)
        #trainer.save_metrics("eval", metrics)
        
    # Predict
    if training_args.do_predict:
        logging.info("*** Predict ***")
        split_name = "test"
        
        start_predicting_time = time.time()
        
        predictions, pred_labels, pred_metrics = trainer.predict(predict_dataset, metric_key_prefix=split_name)
        predictions = np.argmax(predictions, axis=2)
        
        predicting_time = time.time()-start_predicting_time
        predicting_time_str = time.strftime('%H:%M:%S', time.gmtime(predicting_time))
        logging.info(">>>> ⏱ Predictions done in {}.".format(predicting_time_str))
        
        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, pred_labels)
            ]
        flattened_preds = [p for preds in true_predictions for p in preds]
        
        # map predictions on loaded DataFrame
        loaded_df = pd.read_csv(data_args.data_dir+split_name+".tsv", sep="\t")

        sorted_eof_ind = sorted(loaded_df.index[loaded_df["token"]=="EOF"].to_list())
        preds_with_eof = flattened_preds
        for eof_id in sorted_eof_ind:
            preds_with_eof = preds_with_eof[:eof_id]+["EOF"]+preds_with_eof[eof_id:]

        # Put the predictions in <data>
        loaded_df["prediction"] = preds_with_eof
        
        # Break down per file predictions 
        per_file_dict = {}
        for flnm in np.unique(loaded_df[loaded_df["token"]!="EOF"]["file"]):
            file_df = loaded_df[loaded_df["file"]==flnm]

            per_file_dict[flnm] = {'gt_labels': list(file_df["label"])
                                   , 'pred_labels': list(file_df["prediction"])
                                  }
            
        logging.info("📊 Computing performances")
        start_evaluation_time = time.time()
        
        # Run Evaluation
        performances_evaluator = EvaluationChain(eval_args=eval_args,
                                                 ground_truth=list(loaded_df[loaded_df["token"]!="EOF"]["label"]),
                                                 predictions=list(loaded_df[loaded_df["token"]!="EOF"]["prediction"]),
                                                 per_file_labels = per_file_dict,
                                                )
        results_dict = performances_evaluator.compute_scores()
        
        evaluation_time = time.time()-start_evaluation_time
        evaluation_time_str = time.strftime('%H:%M:%S', time.gmtime(evaluation_time))
        logging.info(">>>> ✅ Evaluation done in {}.".format(evaluation_time_str))
        
        # Display results
        logging.info(make_str_results_dict(results_dict, table_name = "~~ PERFORMANCES ON "+split_name.upper()+" SET ~~"))
        logging.info("Storing evaluation results at: {res_path}".format(res_path=eval_args.eval_save_dir))
        # Save results
        performances_evaluator.save_results(prefix=split_name)
        # Store predictions
        if eval_args.save_prediction_files:
            loaded_df.to_csv(path_or_buf=eval_args.eval_save_dir+split_name+".tsv"
                             , sep="\t"
                             , index=False
                             #, header=False
                            )      
    return 
