"""

*** Model Arguments + Parser definitions *** 

"""

import os

import dataclasses 
from dataclasses import dataclass, field
from typing import Optional

from transformers import (
    HfArgumentParser,
    TrainingArguments,
)

from preprocessing.data_utils import *

import copy
    

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%% 📏 Regex %%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

@dataclass
class RegexModelArguments:
    """
    Regex arguments
    """
    regex_type: str = field(
        default="bookNLP",
        metadata={"help": "Type of RegEx model to be used ('bookNLP'|'regexByszuk')"}
    )
    dash_threshold: Optional[float] = field(
        default=None,
        metadata={"help": "Dash threshold for 'bookNLP' regex type."}
    )
    word_tokenizer:str = field(
        default="nltk_tokenization",
        metadata={"help": "Word tokenizer to use ('nltk_tokenization'|'spacy_tokenization')"}
    )

@dataclass
class RegexDataTrainigArguments:
    data_dir: str = field(
        metadata={"help": "Path to the raw `json` corpus."}
    )
    
@dataclass
class RegexDataTrainingArguments:
    output_dir:str = field(
        metadata={"help": "Output directory."}
    )
    do_train:bool = field(
        default=False,
        metadata={"help": "/!\ NOT USED: zero-shot type methods."}
    )
    do_eval:bool = field(
        default=False,
        metadata={"help": "Run method on validation set."}
    )
    do_predict:bool = field(
        default=True,
        metadata={"help": "Run method on test set."}
    )
    
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%% 🔥 Flair %%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
@dataclass
class FlairModelArguments:
    """
    Arguments pertaining to which model/config/embedding are going to be used in Flair implementation.
    """
    embedding_name: str = field(
        metadata={"help": "Embedding from Flair framework (eg. flair, flair_stacked_fasttext, barthez)"}
    )
    max_len: int = field(
        default=256,
        metadata={"help": "Maximum length of the chunks (in terms of word-tokens) fed to the model."}
    )
    batch_len: int = field(
        default=8,
        metadata={"help": "Batches length."}
    )
    max_epochs: int = field(
        default=10,
        metadata={"help": "Maximum of training epochs"}
    )
    do_shuffle: bool = field(
        default=False,
        metadata={"help": "Shuffling examples when training."}
    )
    use_crf: bool = field(
        default=True,
        metadata={"help": "Using a CRF layer."}
    )
    rnn_layers: int = field(
        default=2,
        metadata={"help": "Number of RNN layers."}
    )
    hidden_size: int = field(
        default=256,
        metadata={"help": "Hidden size."}
    )
    
    
    
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%% 🤗 Transformer %%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
        
    max_chunk_len: int = field(
        default=256,
        metadata={"help": "Maximum size (number of word-tokens) per chunk given to the model."},
    )
    
@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task_name: Optional[str] = field(default="ner", metadata={"help": "The name of the task (ner, pos...)."})
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    text_column_name: Optional[str] = field(
        default=None, metadata={"help": "The column name of text to input in the file (a csv or JSON file)."}
    )
    label_column_name: Optional[str] = field(
        default=None, metadata={"help": "The column name of label to input in the file (a csv or JSON file)."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=None,
        metadata={
            "help": "The maximum total input sequence length after tokenization. If set, sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    label_all_tokens: bool = field(
        default=False,
        metadata={
            "help": "Whether to put the label for one word on all tokens of generated by that word or just on the "
            "one (in which case the other tokens will have a padding index)."
        },
    )
    return_entity_level_metrics: bool = field(
        default=False,
        metadata={"help": "Whether to return all the entity levels during evaluation or just the overall ones."},
    )
    reinit_layers: int = field(
        default=0,
        metadata={"help": "re-initialize the last N Transformer blocks."}
    )

    data_dir: str = field(
        default=None,
        metadata={"help": "the data directory"}
    )

    def __post_init__(self):
        if self.dataset_name is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        self.task_name = self.task_name.lower()
        
        
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%% Common arguments %%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

@dataclass
class GeneralArguments:
    model_type: str = field(
        metadata={"help": "Type of model (regex, flair, transformer)."}
    )
    model_natural_name: Optional[str] = field(
        default=None,
        metadata={"help": "Name to refer to the model."}
    )
    
    def __post_init__(self):
        if self.model_natural_name is None:
            self.model_natural_name = self.model_type
            
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%% 📊 Evaluation %%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

@dataclass
class EvalArguments:
    """
    Arguments for evaluation
    """
    
    eval_save_dir: str = field(
        default="results/",
        metadata={"help": "Directory to dump the computed metrics."},
    )
    
    save_prediction_files: bool = field(
        default=False,
        metadata={"help": "Save .tsv files with predictions."},
    )
    
    eval_on_main: bool = field(
        default=True,
        metadata={"help": "Evaluation on main dataset."},
    )
    eval_on_ood: bool = field(
        default=False,
        metadata={"help": "Evaluation on OOD dataset."},
    )
    
    per_file_evaluation: bool = field(
        default=False,
        metadata={"help": "Perform evaluation per file as well."},
    )
    
    aggregation_methods: str = field(
        default=None,
        metadata={"help": "name of numpy methods to aggregate scores computed for each file"},
    )

    token_level_precision_recall_fscore: bool = field(
        default=True,
        metadata={"help": "Compute token-level p, r, f1."},
    )
    strict_match_precision_recall_fscore: bool = field(
        default=True,
        metadata={"help": "Compute SSM p, r, f1."},
    )
    purity_coverage: bool = field(
        default=False,
        metadata={"help": "Compute Purity/Coverage p, c, f1."},
    )
    FairEval_precision_recall_fscore: bool = field(
        default=False,
        metadata={"help": "Compute FairEval p, r, f1."},
    )
    zonemap: bool = field(
        default=False,
        metadata={"help": "Compute Zone Map error."},
    )
        
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%% Parser helpers %%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def string_to_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, type(None)):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ArgumentTypeError(
            f"Truthy value expected: got {v} but expected one of yes/no, true/false, t/f, y/n, 1/0 (case insensitive)."
        )
        
def convert_booleans_from_args(args):
    for arguments in args:
        for field in dataclasses.fields(arguments):
            field_name = field.name
            field_type = field.type
            if field_type==bool or field_type==Optional[bool]:
                setattr(arguments
                        , field_name
                        , string_to_bool(getattr(arguments, field_name))
                       )
    return args
          
"""

Read all configs from config folder and return list of configs

"""

def parse_transformer(config_path):
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, EvalArguments))
    args = parser.parse_json_file(json_file=os.path.join(config_path))
    
    model_args, data_args, training_args, eval_args = convert_booleans_from_args(args)
    
    return model_args, data_args, training_args, eval_args

def parse_flair(config_path):
    parser = HfArgumentParser((FlairModelArguments, DataTrainingArguments, TrainingArguments, EvalArguments))
    args = parser.parse_json_file(json_file=os.path.join(config_path))
    
    model_args, data_args, training_args, eval_args = convert_booleans_from_args(args)
    
    return model_args, data_args, training_args, eval_args

def parse_regex(config_path):
    parser = HfArgumentParser((RegexModelArguments
                               , RegexDataTrainigArguments
                               , RegexDataTrainingArguments
                               , EvalArguments
                              ))
    args = parser.parse_json_file(json_file=os.path.join(config_path))
    
    model_args, data_args, training_args, eval_args = convert_booleans_from_args(args)
    
    return model_args, data_args, training_args, eval_args

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%% Main methods %%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def read_all_configs(path_config_folder:str="experiments_configs",
                     configs_list:list = []
                    )->list:
    """
    Read all/selected json configuration files in a folder
    and return a list of list containing the parsed arguments.
    
    /!\ The configuration files' names must start with one of:
        - 'regex_'
        - 'flair_'
        - 'transformer_'
        which defines the parser to be called.
        AND arguments within the file must comply with the prefix,
            ie. 'regex_config.json' must contain arguments to launch regex experiment.
        
    Arguments:
        path_config_folder (str: default = "experiments_configs" --> here)
            path to the folder containing the json configs files.
        configs_list (list: default=[])
            list of .json files' names (in <path_config_folder>) to be parsed
            - [if <configs_list> empty] => parse all .json files in <path_config_folder>
            
    Returns:
        (list) list of list containing parsed arguments used to instantiate (W)DSR models
            [[general_args, model_args, data_args, training_args, eval_args], ...]
              |                  |          |             |           |
        General              Model     Data training    Training      Evaluation arguments
      arguments          arguments       arguments      arguments
    """
    
    # 1. retrieve all config .json files
    if len(configs_list)<1:
        for file in os.listdir(path_config_folder):
            if file.endswith(".json"):
                configs_list += [file]#os.path.join("../", file))
            
    # 2. parse each of the config .json files
    arguments_list = []
    for file in configs_list:
        # 2.1) determine type of model ==> arguments parsed
        type_of_model = file.split("_")[0]
        assert type_of_model in ["regex", "flair", "transformer"], "Parser not defined for {}.".format(file)
        
        # 2.2) parse config file accordingly
        if type_of_model=="regex": 
            model_args, data_args, training_args, eval_args = parse_regex(os.path.join(path_config_folder, file))
            
        elif type_of_model=="flair":
            model_args, data_args, training_args, eval_args = parse_flair(os.path.join(path_config_folder, file))
            
        elif type_of_model=="transformer":
            model_args, data_args, training_args, eval_args = parse_transformer(os.path.join(path_config_folder, file))
        
        general_args = GeneralArguments(model_type=type_of_model)
        
        arguments_list += [(general_args, model_args, data_args, training_args, eval_args)]
    
    # 3. return list of parsed arguments
    return arguments_list

def load_base_vary_param(path_base_config_folder:str,
                         base_config_name:str,
                         param_name_to_vary:str,
                         alternative_params:list
                        )->list:
    """
    return list of arguments based on an original config with one parameter varying
    """
    args = read_all_configs(path_config_folder = path_base_config_folder,
                            configs_list = [base_config_name]
                           )
    original_arguments = args[0]
    
    
    # Retrieve the index in the original arguments list in which the parameter appear
    found_param = False
    i_arg = 0
    while (not found_param and i_arg<len(original_arguments)):
        curr_attrs = vars(original_arguments[i_arg])
        if param_name_to_vary in curr_attrs.keys():
            found_param = True
        else:
            i_arg += 1
    # If not found --> Raise error: not possible to make it vary
    if not found_param:
        raise ValueError("{key} not found in configuration list {flnm} at {path}".format(key=param_name_to_vary
                                                                                         , flnm=base_config_name
                                                                                         , path=path_base_config_folder
                                                                                        ))
        
    list_alt_args = []
    for param_alt in alternative_params:
        alt_args = copy.deepcopy(original_arguments)
        setattr(alt_args[i_arg], param_name_to_vary, param_alt)
        list_alt_args += [alt_args]
        
    return list_alt_args


def loaded_config_card(args_list:list
                       , card_width:int=100
                       , header_name:str="== LOADED CONFIG ==" 
                      )->str:
    """
    Make a displayable string to recap loaded arguments.
    """
    name_width = name_width = int(card_width/2)-1
    
    wsb = " "*(int((card_width-len(header_name)-2)/2))
    wsa = " "*(card_width-2-len(wsb+header_name))

    midline = "+{}+".format("-"*(card_width-2),)

    str_card = "\n{upper_line}\n|{ws1}{name}{ws2}|\n{upper_line}\n".format(upper_line=midline,
                                                                           ws1=wsb,
                                                                           ws2=wsa,
                                                                           name=header_name
                                                                          )
    
    for args in args_list:
        args_attrs = vars(args)
        for k, val in zip(args_attrs.keys(), args_attrs.values()):
            kname = " - "+k
            wsmid = " "*(name_width-len(kname))
            wsend = " "*(card_width-2-(len(kname)+len(wsmid)+len(str(val))))
            
            if (card_width-2-(len(kname)+len(wsmid)+len(str(val))))<0:
                val1 = val[:card_width-2-(len(kname)+len(wsmid))]
                val2 = val[card_width-2-(len(kname)+len(wsmid)):]
                val2 += " "*(card_width-2-(len(kname)+len(wsmid)+len(val2)))
            else: 
                val1=val
                val2=None
            
            new_line = "|{name}{ws1}{arg}{ws2}|\n".format(name=kname, arg=val1,
                                                          ws1=wsmid,
                                                          ws2=wsend
                                                         )
            if val2:
                new_line += "|{wsname}{ws1}{arg2}|\n".format(wsname=" "*len(kname)
                                                             , ws1=wsmid
                                                             , arg2=val2
                                                            )
            str_card+=new_line
        str_card += midline+"\n"
        
    return str_card