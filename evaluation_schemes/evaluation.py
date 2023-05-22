"""
*** Evaluation module ***

Comprising:

I - Class definitions

    A) Evaluator
    B) EvaluationChain

II - Evaluation methods

    A) Token-level evaluation
    B) Strict Sequence Match Evaluation
    C) Purity/Coverage
    D) FairEval
    E) Zone Map Error
    (F) Partial Sequence Match)

    By default each of the evaluation scores can be computed by giving
        `gt_labels` : list of binary ground truth labels ('O' | 'DS')
        `pred_labels` : list of binary predicted labels ('O' | 'DS')
        [both must be coherent (same length)]

        --> The output is a `dict` with string keys referring to the computed scores
            and their value
"""

import itertools
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import ast
import warnings

import json
import os
import logging

from typing import Optional

# UTILS
#from evaluation_schemes.eval_utils import *
from preprocessing.utils_converters import *

# Scikit-learn
import sklearn.metrics as sk_eval

# seqeval
import seqeval.metrics as seq_eval
from seqeval.scheme import IOB1, IOB2, IOE1, IOE2, IOBES, BILOU

# pyannote
from pyannote.core import Annotation, Segment, Timeline

from pyannote.metrics.segmentation import SegmentationPurity
from pyannote.metrics.segmentation import SegmentationCoverage
from pyannote.metrics.segmentation import SegmentationPrecision
from pyannote.metrics.segmentation import SegmentationRecall

segmentation_purity = SegmentationPurity()
segmentation_coverage = SegmentationCoverage()
segmentation_precision = SegmentationPrecision()
segmentation_recall = SegmentationRecall()

from pyannote.metrics.diarization import DiarizationPurity
from pyannote.metrics.diarization import DiarizationCoverage
from pyannote.metrics.diarization import DiarizationErrorRate

diarization_purity = DiarizationPurity()
diarization_coverage = DiarizationCoverage()
diarization_der = DiarizationErrorRate()

# Fair Eval
from evaluation_schemes.FairEval import *

""" 

I - Define classes to be instantiated for Evaluation 

"""

class Evaluator:
    """
    Object containing an evaluation method able to compute performance scores
    by comparing binary classification predictions with ground truth labels
    
    TODO:
    - [ ] DOCUMENTATION 
    - [ ] pass arguments from class to _evaluate method
    """
    def __init__(self,
                 e_name,
                 e_method,
                 **kwargs
                ):
        self.__name__ = e_name
        self.eval_method = e_method#lambda x, y : e_method(x,y, **kwargs)

    def _evaluate(self
                  , gt_labels
                  , pred_labels
                 )->dict:

        return self.eval_method(gt_labels
                                , pred_labels
                               )

class EvaluationChain:
    """
    Evaluation instantiated with parsed EvalArguments.
    Based on a list of Evaluator objects, compute the scores associated with gt_labels and pred_labels.
    
    TODO:
    - [ ] DOCUMENTATION 
    - [ ] Typing
    """
    def __init__(self,
                 # replace everything with eval_args
                 eval_args,
                 ground_truth=[],
                 predictions=[],
                 per_file_labels={},
                 # aggregation_methods: dict= {'name': method}
                ):
        
        # store arguments
        self.eval_args = eval_args
        
        self.save_dir = eval_args.eval_save_dir
        
        evaluator_list = []
        
        if eval_args.token_level_precision_recall_fscore: 
            evaluator_list+=[Evaluator(e_name = "TokenLevel",
                                       e_method = token_level_precision_recall_fscore
                                      )
                            ]
        if eval_args.strict_match_precision_recall_fscore: 
            evaluator_list+=[Evaluator(e_name = "SSM",
                                       e_method = strict_match_precision_recall_fscore
                                      )
                            ]
        if eval_args.purity_coverage: 
            evaluator_list+=[Evaluator(e_name = "PurCov",
                                       e_method = purity_coverage
                                      )
                            ]
        if eval_args.FairEval_precision_recall_fscore: 
            evaluator_list+=[Evaluator(e_name = "FairEval",
                                       e_method = FairEval_precision_recall_fscore
                                      )
                            ]
        if eval_args.zonemap: 
            evaluator_list+=[Evaluator(e_name = "ZoneMap",
                                       e_method = zonemap
                                      )
                            ]
        
        self.EvaluatorList = evaluator_list
        
        if (not ground_truth) and (not predictions) and per_file_labels:
            # if there are no overall results but still per file labels given
            # then create the liste of overall GT and predictions by appending
            # the individual GT/preds from each of the file
            ground_truth = []
            predictions = []
            for f_val in per_file_labels.values():
                ground_truth += f_val["gt_labels"]
                predictions += f_val["pred_labels"]
        
        # SANITY CHECKS
        if len(ground_truth)!=len(predictions):
            raise ValueError("Ground truth and prediction labels' list should have the same length. (GT: {gt_len} ; Pred: {pred_len})".format(gt_len=len(ground_truth), pred_len=len(predictions)))
            
        self.gt_labels = ground_truth
        self.pred_labels = predictions
        
        self.EvaluationPerFile = eval_args.per_file_evaluation
        self.filesToEvaluate = per_file_labels
        
        if self.EvaluationPerFile and eval_args.aggregation_methods:
            
            agg_list = ast.literal_eval(eval_args.aggregation_methods)
            self.AggregationMethods = {}
            for agg in agg_list:
                # get methods corresponding to their string name in numpy 
                self.AggregationMethods.update({agg: getattr(np, agg)})
                
                self.ResultsPerfile = {}
    
    def compute_scores(self
                      )->dict:
        
        # 0. Instantiate the `dict` meant to store computed scores
        scores_dict = {}
        
        # 1. Iterate over each Evaluator in the EvaluatorList
        for evaluator in self.EvaluatorList:
            # 1.1 Compute the current Evaluator scores
            eval_scores = evaluator._evaluate(self.gt_labels
                                              , self.pred_labels
                                             )
            # 1.2 Append the resulting metrics in the `dict`
            scores_dict.update(eval_scores)
            
        # 2. [if EvaluationPerFile == True] => Run per file evaluation 
        if self.EvaluationPerFile and self.filesToEvaluate:
            scores_dict.update(self.per_file_evaluation())
            
        # 3. Store the computed metrics in the field <results>
        self.results = scores_dict
        
        return self.results
    
    def per_file_evaluation(self)->dict:
        
        # 0. Instantiate the `dict` meant to store computed scores for each file as an object field <resultsPerFile>
        #.   --> {flnm: {metric_name1: score1, metric_name2:score2, ...}, ...}
        self.resultsPerFile = {}
        
        # 1. Iterate over each (key, value) pair of given <filesToEvaluate>
        for k, val in zip(self.filesToEvaluate.keys(), self.filesToEvaluate.values()):
            # 1.1 Instantiate `dict` to store metrics for the current file (GT/pred pair)
            file_scores = {}
            # 1.2 Compute the metrics on the GT/pred pair for each Evaluator from EvaluatorList
            for evaluator in self.EvaluatorList:
                eval_scores = evaluator._evaluate(val["gt_labels"]
                                                  , val["pred_labels"]
                                                 )
                file_scores.update(eval_scores)
            self.resultsPerFile[k] = file_scores
                    
        # 2. Aggregate the results over the files
        # 2.1 Retrieve computed metrics' names (that needs to be aggregated)
        computed_scores = list(file_scores.keys())
        agg_results = {}
        # 2.2 Iterate over each computed metric
        for score_name in computed_scores:
            # 2.2.1 Retrieve the associated score for each file --> make a list with all of them
            per_file_scores = [res_file[score_name] for res_file in self.resultsPerFile.values()]
            # 2.2.2 Aggregate these scores with all the AggregationMethods
            for method_name, method in zip(self.AggregationMethods.keys(), self.AggregationMethods.values()):
                agg_results["{s}_{m}".format(s=score_name,m=method_name)] = method(per_file_scores)
                
        return agg_results
    
    def save_results(self
                     , prefix=""
                    ):
        """
        Save the computed metrics `dict` as a json file
        """
        try:
            results_to_save = self.results
        except:
            warnings.warn("EvaluatorChain <results> field is empty. Nothing will be stored.")
            return
                
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            
        results_file_name = prefix+"_performances.json" if prefix else "performances.json"
        
        with open(os.path.join(self.save_dir, results_file_name), 'w+') as fp:
            json.dump(results_to_save, fp)
        fp.close()
        
        return
    
    

""" 

II - Implement evaluation methods to be used

Inputs: 
    - gt_labels:(list)
        List of binary ground truth labels        
    - pred_labels:(list)
        List of binary prediction labels        
    - additional arguments
Output: 
    - (dict): { 'score_name_A': computed_score_A
                'score_name_B': computed_score_B
                ...
              }
"""

## A) Token-level

def token_level_precision_recall_fscore(gt_labels:list,
                                        pred_labels:list,
                                        pos_label="DS"
                                       )->dict:
    """
    Compute binary precision, recall, F1-score from list of GT/prediction labels
    Based on scikit-learn: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html
    
    Arguments:
        gt_labels (list)
            List of Ground Truth labels 
        pred_labels (list)
            List of prediction labels
        --> If BIO/BIOES/... schemes are used, labels will be converted to binary class labels
            
        pos_label (: "DS")
            Positive Label used to compute the binary TP/FP/FN yielding precision, recall, F1
    Returns:
        (dict): precision, recall, f1
        
    Example:
    token_level_precision_recall_fscore(['S-DS', 'O', 'B-DS', 'I-DS', 'E-DS', 'O'],
                                        ['O'   , 'O', 'B-DS', 'I-DS', 'E-DS', 'O'],
                                        )
    >>> {
            'tokenLevel_precision': 1.0,
            'tokenLevel_recall': 0.75,
            'tokenLevel_f1': 0.8571428571428571
        }
    """

    # if BIO --> convert to binary
    used_labels = set(gt_labels).union(set(pred_labels))
    gt_labels = clean_labels_from_prefixes(labels_list=gt_labels,
                                           used_labels=used_labels
                                          )
    pred_labels = clean_labels_from_prefixes(labels_list=pred_labels,
                                           used_labels=used_labels
                                          )
        
    # Compute scores with sklearn.metrics 
    p, r, f, _ = sk_eval.precision_recall_fscore_support(gt_labels
                                                         , pred_labels
                                                         , pos_label=pos_label
                                                         , average="binary"
                                                        )
    
    return {
        "tokenLevel_precision":p,
        "tokenLevel_recall":r,
        "tokenLevel_f1":f,
    }

## B) Strict sequence match

def strict_match_precision_recall_fscore(gt_labels:list
                                         , pred_labels:list
                                         , eval_scheme=IOB2
                                         , mode:str="strict"
                                         , label_scheme:str="IO"
                                         , pos_label="DS"
                                        )->dict:
    """
    Compute strict sequences evaluation scores: precision, recall and F1-score
    Based on seqeval: https://github.com/chakki-works/seqeval
    
    Arguments:
        gt_labels (list)
            Ground Truth annotations respecting given <eval_scheme> conventions
        pred_labels (list)
            Model prediction respecting given <eval_scheme> conventions
            — must to correspond to <gt_labels> 
        eval_scheme (one of: IOBES, IOB2, IOE2)
            Convention for the evaluation scheme => determines the evaluation scheme
        mode (str: "strict")
            Mode for seqeval evaluation
        label_scheme (str: "IO")
            Annotations scheme used in the input lists (GT/preds)
        pos_label (: "DS")
            Positive Label used to compute the scores
    Returns:
        (dict): precision, recall, f1
        
    Example:
    strict_match_precision_recall_fscore(['S-DS', 'O', 'B-DS', 'I-DS', 'E-DS', 'O'],
                                         ['O'   , 'O', 'B-DS', 'I-DS', 'E-DS', 'O'],
                                         eval_scheme=IOBES
                                        )
    >>> {
            'SSM_precision': 1.0,
            'SSM_recall': 0.5,
            'SSM_f1': 0.6666666666666666
        }
    """
    
    # Convert Binary list of inputs to desired evaluation scheme
    # OR input annotation scheme different to desired evaluation scheme
    eval_scheme_str = eval_scheme.__name__
    if label_scheme=="IO" or label_scheme!=eval_scheme_str:
        # /!\ Implemented schemes are "IOBES", "IOB2", "IOE2"
        gt_labels = span_to_bio(tok_spans=[]
                                , tok_list=[pos_label in lab for lab in gt_labels]
                                , scheme = eval_scheme_str
                                , from_boolean_list = True
                               )
        pred_labels = span_to_bio(tok_spans=[]
                                  , tok_list=[pos_label in lab for lab in pred_labels]
                                  , scheme = eval_scheme_str
                                  , from_boolean_list = True
                                 )
    
    # List of list for to be seq_eval compliant
    gt_seq = [gt_labels]
    pred_seq = [pred_labels]
        
    # compute scores with seq_eval
    p_seqeval = seq_eval.precision_score(gt_seq, pred_seq, scheme=eval_scheme, mode=mode)
    r_seqeval = seq_eval.recall_score(gt_seq, pred_seq, scheme=eval_scheme, mode=mode)
    f_seqeval = seq_eval.f1_score(gt_seq, pred_seq, scheme=eval_scheme, mode=mode)
    
    return {
        "SSM_precision":p_seqeval,
        "SSM_recall":r_seqeval,
        "SSM_f1":f_seqeval
    }

## C) Purity/Coverage

def purity_coverage(gt_labels:list
                    , pred_labels:list
                    , only_pos:bool=False
                    , pos_label="DS"
                    , tolerance_window:int=0
                   )->dict:
    
    """
    Compute Purity, Coverage and corresponding F1 like score.
    Based on: pyannote.metrics
    
    Arguments:
        gt_labels (list)
            Ground Truth annotations
            — if not binary (ie. with IOB/IOBES/... schemes), will be converted to binary
        pred_seq (list)
            — must to correspond to <gt_labels>
            — if not binary (ie. with IOB/IOBES/... schemes), will be converted to binary
        only_pos (bool: False)
            Consider only segments with positive labels
        mode (str: "strict")
            Mode for seqeval evaluation
        label_scheme (str: "IO")
            Annotations scheme used in the input lists (GT/preds)
        pos_label (: "DS")
            Positive Label used to compute the scores
        tolerance_window (int:0)
            tolerance window (in tokens here/list indices) to match segments positively
    Returns:
        (dict): purity, coverage, f1
        
    Example:
    purity_coverage(['S-DS', 'O', 'B-DS', 'I-DS', 'E-DS', 'O'],
                    ['O'   , 'O', 'B-DS', 'I-DS', 'E-DS', 'O'],
                    )
    >>> {
            'PC_purity': 0.8333333333333334,
            'PC_coverage': 1.0,
            'PC_f1': 0.9090909090909091
        }
    """
    
    # if BIO --> convert to binary
    used_labels = set(gt_labels).union(set(pred_labels))
    gt_labels = clean_labels_from_prefixes(labels_list=gt_labels,
                                           used_labels=used_labels
                                          )
    pred_labels = clean_labels_from_prefixes(labels_list=pred_labels,
                                           used_labels=used_labels
                                          )
    
    # Set objects to compute metrics using pyannote.metrics
    uem = Timeline([Segment(0, len(gt_labels))])
    
    segments_gt = [(Segment(g[0][0], g[0][0]+len(g)), key)
                   for key, group in itertools.groupby(enumerate(gt_labels), key=lambda v: v[1])
                   for g in (list(group),)
                  ]    
    annot_gt = Annotation()
    for seg, lab in segments_gt:
        if not only_pos or lab==pos_label:
            annot_gt[seg] = lab
            
    segments_pred = [(Segment(g[0][0], g[0][0]+len(g)), key)
                     for key, group in itertools.groupby(enumerate(pred_labels), key=lambda v: v[1])
                     for g in (list(group),)
                    ]
    annot_pred = Annotation()
    for seg, lab in segments_pred:
        if not only_pos or lab==pos_label:
            annot_pred[seg] = lab
    
    # Compute metrics
    purity = segmentation_purity(annot_gt
                                 , annot_pred
                                 , tolerance=tolerance_window
                                 , uem=uem
                                )
    coverage = segmentation_coverage(annot_gt
                                     , annot_pred
                                     , tolerance=tolerance_window
                                     , uem=uem
                                    )
    
    if (purity+coverage)==0:
        fmeasure = 0
    else:
        fmeasure = 2*purity*coverage/(purity+coverage)
    
    return {
        "PC_purity":purity,
        "PC_coverage":coverage,
        "PC_f1":fmeasure
    }

## D) Fair Eval

def FairEval_precision_recall_fscore(gt_labels:list
                                     , pred_labels:list
                                     , eval_weights:dict=None
                                     , pos_label:str="DS"
                                     , return_counts:bool = False
                                     , label_scheme="IO"
                                    )->dict:
    """
    Compute precision, recall and F1-score
    based on Ortmann FairEval framework and code: 
        https://github.com/katrinortmann/FairEval
        @ commit 606a031
        
    Arguments:
        gt_spans (list)
            Ground Truth annotations 
        pred_spans (list)
            Model prediction of binary labels
        eval_weights (dict : None)
            Mitigating parameter Classification / Segmentation 
            — if None, values proposed in by Ortmann are used :
                {
                    "TP" : {"TP" : 1},
                    "FP" : {"FP" : 1},
                    "FN" : {"FN" : 1},
                    "LE" : {"TP" : 0, "FP" : 0.5, "FN" : 0.5},
                    "BE" : {"TP" : 0, "FP" : 0.5, "FN" : 0.5},
                    "LBE" : {"TP" : 0, "FP" : 0.5, "FN" : 0.5}
                }
        pos_label (:"DS")
            positive label
        return_counts (bool: False)
            Return as well a dict containing the decomposition of the error along the typology
            (TP, FP, FN, LE [X], LBE [X], BE, BEO, BES, BEL)
        
        
    Returns:
        (dict): precision, recall, F1
        
    Example:
    FairEval_precision_recall_fscore(['S-DS', 'O', 'B-DS', 'I-DS', 'E-DS', 'O'],
                                     ['O'   , 'O', 'B-DS', 'I-DS', 'E-DS', 'O'],
                                    )
    >>> {
            'FE_precision': 1.0,
            'FE_recall': 0.5,
            'FE_f1': 0.6666666666666666
        }
    """
    
    if label_scheme!="IOB2":
        # /!\ Implemented schemes are "IOBES", "IOB2", "IOE2"
        gt_labels = span_to_bio(tok_spans=[]
                                , tok_list=[pos_label in lab for lab in gt_labels]
                                , scheme = "IOB2"
                                , from_boolean_list = True
                               )
        pred_labels = span_to_bio(tok_spans=[]
                                  , tok_list=[pos_label in lab for lab in pred_labels]
                                  , scheme = "IOB2"
                                  , from_boolean_list = True
                                 )

    # convert spans to FairEval-friendly format
    def bio_to_word_spans(bio_list):
        word_spans_list = []

        in_quote = False
        curr_start = None

        for word_id, lab in enumerate(bio_list):
            if lab!="O" and (not in_quote):
                in_quote = True
                curr_start = word_id
            elif in_quote and (lab=="O" or lab=="B-DS"):
                curr_end = word_id-1
                word_spans_list += [(curr_start, curr_end)]
                if lab=="O":
                    in_quote=False
                else:
                    in_quote=True
                    curr_start=word_id
        if in_quote:
            word_spans_list += [(curr_start, word_id)]
            
        return word_spans_list
    
    gt_spans = bio_to_word_spans(gt_labels)
    pred_spans = bio_to_word_spans(pred_labels)
    
    gt_FE = [[pos_label, s[0], s[1]-1, set(np.arange(s[0],s[1]))] for s in gt_spans]
    pred_FE = [[pos_label, s[0], s[1]-1, set(np.arange(s[0],s[1]))] for s in pred_spans]
    
    res_FairEval = compare_spans(gt_FE
                                 , pred_FE
                                 #, focus
                                )
    fair_eval_dict = res_FairEval["per_label"]["fair"][pos_label]
    trad_eval_dict = res_FairEval["per_label"]["traditional"][pos_label]
    
    if eval_weights:
        p_FE = precision(fair_eval_dict, version="weights", weights=eval_weights)
        r_FE = recall(fair_eval_dict, version="weights", weights=eval_weights)
    else:
        p_FE = precision(fair_eval_dict, version="fair")
        r_FE = recall(fair_eval_dict, version="fair")
        
    if (p_FE+r_FE)>0:
        f_FE = 2*(p_FE*r_FE)/(p_FE+r_FE)
    else:
        f_FE = 0.
    
    if return_counts:
        return {
            "FE_precision":p_FE,
            "FE_recall":r_FE,
            "FE_f1":f_FE
        }, (fair_eval_dict, trad_eval_dict)
    else:
        return {
            "FE_precision":p_FE,
            "FE_recall":r_FE,
            "FE_f1":f_FE
        }
    
## E) Zone Map Error

def zonemap(gt_labels:list
            , pred_labels:list
            , alpha_c:np.float64=0.
            , alpha_ms:np.float64=0.5
            , dist_classes:np.float64=0
            , pos_lab:str="DS"
            , spans_input:bool=False
            , inspect:bool=False
            , return_groups:bool=False
           )->dict:
    """
    Compute error as inspired by ZoneMap Error by Galibert, 2014
        
    Arguments:
        gt_labels (list)
            Ground Truth annotations 
            if <spans_input> == True : spans of positive sequences
            if <spans_input> == False : binary labels (including <pos_label>)
        pred_labels (list)
            Model prediction of binary labels (including <pos_label>)
            — must to correspond to <gt_labels> 
        alpha_c (float: 0.) /!\ NOT USED /!\
            Mitigating parameter Classification / Segmentation
        alpha_ms (float: 0.) 
            Mitigating parameter for Merge & Split sub-zone errors
        dist_classes (float: 0.) /!\ NOT USED /!\
            Mitigating parameter Classification / Segmentation 
        pos_label (:"DS")
            positive label
        spans_input (bool: False)
            Is the input already a list of spans ?
        inspect (bool: False)
            Show the plot corresponding to the computation
        return_groups (bool: False)
            Return as well a dict containing the decomposition of the error along the typology
        
        
    Returns:
        (dict): E_ZM
        
        if <return_groups> == True : E_ZM, (dict): groups
        
    Examples:
    zonemap(['O', 'O', 'O', 'O', 'DS', 'DS', 'DS'],
            ['DS', 'DS', 'DS', 'O', 'DS', 'DS', 'O'],
            pos_label = "DS",
            alpha_c = 0,
            alpha_ms = 0.5,
            return_groups = True
            )
    >>> ( 1.3333333333333333 ,
          { 1: {'gt': [0], 'pred': [1], 'type': 'Match', 'error': 1.0},
            2: {'gt': [], 'pred': [0], 'type': 'FalseAlarm', 'error': 3.0}
            }
        )
    """
    
    # if BIO --> convert to binary
    used_labels = set(gt_labels).union(set(pred_labels))
    gt_labels = clean_labels_from_prefixes(labels_list=gt_labels,
                                           used_labels=used_labels
                                          )
    pred_labels = clean_labels_from_prefixes(labels_list=pred_labels,
                                           used_labels=used_labels
                                          )
    
    # instanciations
    if not spans_input:
        boolean_gt = [lab==pos_lab for lab in gt_labels]
        boolean_pred = [lab==pos_lab for lab in pred_labels]
        gt_spans = [(g[0][0], g[0][0]+len(g))
                    for i, (key, group) in enumerate(itertools.groupby(enumerate(boolean_gt), key=lambda v: v[1]))
                    if key
                    for g in (list(group),)
                   ]
        pred_spans = [(g[0][0], g[0][0]+len(g))
                      for i, (key, group) in enumerate(itertools.groupby(enumerate(boolean_pred), key=lambda v: v[1]))
                      if key
                      for g in (list(group),)
                     ]
    else:
        gt_spans = gt_labels
        pred_spans = pred_labels
        
    
    link_forces = []
    links_matrix = np.zeros(shape=(len(gt_spans),len(pred_spans)))
    
    
    # 1. Computing link forces
    for i, gt_s in enumerate(gt_spans):
        gt_start = gt_s[0]
        gt_end = gt_s[1]
        gt_l = gt_end-gt_start
        for j, pred_s in enumerate(pred_spans):
            pred_start = pred_s[0]
            pred_end = pred_s[1]
            pred_l = pred_end-pred_start
            
            overlap = np.min([gt_end, pred_end]) - np.max([gt_start, pred_start])
            overlap = overlap if overlap>0 else 0
            
            link_force = overlap**2/gt_l**2+overlap**2/pred_l**2
            links_matrix[i][j] = link_force
            link_forces += [(link_force, (i,j))]
                        
    sorted_links = sorted(link_forces, key=lambda f: f[0])[::-1]
    sorted_links = [f for f in sorted_links if f[0]>0]
    
    # 2. Zone Grouping & Configurations type
    # group_i = {gt:[X], pred:[X'], type:_type_ }
    pred_zones_grouped = {} #key=index ; values=group index
    gt_zones_grouped = {}
    groups = {}
    k_group = 1
    
    for l_ij in sorted_links:
        gt_i = l_ij[1][0]
        pred_j = l_ij[1][1]
        
        if not (gt_i in gt_zones_grouped.keys()):
            if not (pred_j in pred_zones_grouped.keys()):
                groups[k_group] = {"gt":[gt_i]
                                   , "pred":[pred_j]
                                   , "type":"Match"
                                  }
                pred_zones_grouped[pred_j] = k_group
                gt_zones_grouped[gt_i] = k_group
                k_group += 1
                
            else:
                group_pred_j = pred_zones_grouped[pred_j]
                if len(groups[group_pred_j]["pred"])==1:
                    groups[group_pred_j] = {"gt":groups[group_pred_j]["gt"]+[gt_i]
                                            , "pred":groups[group_pred_j]["pred"]
                                            , "type": "Merge"
                                           }
                    gt_zones_grouped[gt_i] = group_pred_j
        else:
            group_gt_i = gt_zones_grouped[gt_i]
            if len(groups[group_gt_i]["gt"])==1:
                groups[group_gt_i] = {"gt":groups[group_gt_i]["gt"]
                                      , "pred":groups[group_gt_i]["pred"]+[pred_j]
                                      , "type": "Split"
                                     }
                pred_zones_grouped[pred_j] = group_gt_i
                
    
    
    # remaining spans
    for i, gt_s in enumerate(gt_spans):
        if not (i in gt_zones_grouped.keys()):
            groups[k_group] = {"gt":[i]
                               , "pred":[]
                               , "type":"Miss"
                              }
            k_group +=1
    for j, pred_s in enumerate(pred_spans):
        if not (j in pred_zones_grouped.keys()):
            groups[k_group] = {"gt":[]
                               , "pred":[j]
                               , "type":"FalseAlarm"
                              }
            k_group +=1
            
    # 3. Comuting Error
    errors_s = []
    errors_c = []
    errors_z = []
    for g in groups.values():
        # Simplifying hypothesis: NO TAKING INTO ACCOUNT CLASSIFICATION ERROR <=> alpha_c = 0
        e_s, e_c, e_z = 0, 0, 0
        if g["type"]=="FalseAlarm":
            e_s = pred_spans[g["pred"][0]][1]-pred_spans[g["pred"][0]][0]
            e_c = e_s
            
        elif g["type"]=="Miss":
            e_s = gt_spans[g["gt"][0]][1]-gt_spans[g["gt"][0]][0]
            e_c = e_s
        
        elif g["type"]=="Match":
            # Only 1 GT and only 1 pred
            gt_min = gt_spans[g["gt"][0]][0]
            pred_min = pred_spans[g["pred"][0]][0]
            gt_max = gt_spans[g["gt"][0]][1]
            pred_max = pred_spans[g["pred"][0]][1]
            
            g_min = np.min([gt_min, pred_min])
            g_max = np.max([gt_max, pred_max])
            len_union = g_max-g_min
            
            len_overlap = np.min([gt_max, pred_max]) - np.max([gt_min, pred_min])
            len_overlap = len_overlap if len_overlap>0 else 0 # should not be
            
            e_s = len_union-len_overlap
            e_c = dist_classes*len_overlap+e_s
            
        # !! MODIFIED VERSIONS FOR SPLIT AND MERGE !! 
        ## ==> error should grow with: 
        ### -> _non_overlapping_length_
        ### -> _number_of_segments_
        ### -> _merge_split_coefficient_
        ## ==> caught zones should not be penalized more than if missed
        elif g["type"]=="Split":
            # Only 1 GT and >1 pred
            gt_min = gt_spans[g["gt"][0]][0]
            gt_max = gt_spans[g["gt"][0]][1]
            
            pred_min = np.min([pred_spans[pred_id][0] for pred_id in g["pred"]])
            pred_max = np.max([pred_spans[pred_id][1] for pred_id in g["pred"]])
            
            g_min = np.min([gt_min, pred_min])
            g_max = np.max([gt_max, pred_max])
            len_union = g_max-g_min
            
            overlaps = [np.min([gt_max, pred_spans[pred_id][1]]) - np.max([gt_min, pred_spans[pred_id][0]])
                        for pred_id in g["pred"]
                       ]
            len_overlap = np.sum([o for o in overlaps if o>0])
            
            overlaps_not_largest = sorted(overlaps)[:-1]
            len_overlap_not_largest = np.sum([o for o in overlaps_not_largest if o>0])
            
            e_s_miss_fa = (len_union-len_overlap)
            e_s_seg_error = alpha_ms*len_overlap_not_largest*((len(g["pred"])-1)/len(g["pred"]))
            # if many small pieces => Error grows
            # else: less error as alpha_ms mitigates the term compared to Miss or FA
            
            e_s = e_s_miss_fa + e_s_seg_error
            #e_s = (len_union-len_overlap)+alpha_ms*len(g["pred"]) # WHY ERROR proportional to OVERLAP ??
            e_c = e_s_miss_fa + (len(g["pred"])-1+dist_classes)*len_overlap
            
        elif g["type"]=="Merge":
            # >1 GT and only 1 pred
            gt_min = np.min([gt_spans[gt_id][0] for gt_id in g["gt"]])
            gt_max = np.max([gt_spans[gt_id][1] for gt_id in g["gt"]])
            
            pred_min = pred_spans[g["pred"][0]][0]
            pred_max = pred_spans[g["pred"][0]][1]
            
            g_min = np.min([gt_min, pred_min])
            g_max = np.max([gt_max, pred_max])
            len_union = g_max-g_min
            
            overlaps = [np.min([pred_max, gt_spans[gt_id][1]]) - np.max([pred_min, gt_spans[gt_id][0]])
                        for gt_id in g["gt"]
                       ]
            len_overlap = np.sum([o for o in overlaps if o>0])
            
            overlaps_not_largest = sorted(overlaps)[:-1]
            len_overlap_not_largest = np.sum([o for o in overlaps_not_largest if o>0])
            
            e_s_miss_fa = (len_union-len_overlap)
            e_s_seg_error = alpha_ms*len_overlap_not_largest*((len(g["gt"])-1)/len(g["gt"]))#(len(g["gt"])-1)
            
            e_s = e_s_miss_fa + e_s_seg_error
            e_c = e_s_miss_fa + (len(g["pred"])-1+dist_classes)*len_overlap
            
            
        e_z = (1-alpha_c)*e_s+alpha_c*e_c
        errors_z+= [e_z]
        errors_s+= [e_s]
        errors_c+= [e_c]
        
        g["error"]=e_z
        
    # visualize link forces between GT and pred
    # + different groups
    if inspect:
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,5))
        sns.heatmap(links_matrix
                    , ax=ax[0]
                    , vmin=0
                    , vmax=2
                   )
        
        groups_types = ["Miss", "FalseAlarm", "Split", "Merge"]
        x_axis = np.arange(len(groups_types))
        tot_gt = np.sum([gt_s[1]-gt_s[0] for gt_s in gt_spans])
        for t_i, t in enumerate(groups_types):
            type_error = np.sum([g["error"] for g in groups.values() if g["type"]==t])/tot_gt
            nb_type = np.sum([g["type"]==t for g in groups.values()])
            ax[1].bar(t_i, nb_type, label=r"$E_{Z} = $"+str(type_error))
        
        ax[1].legend()
        ax[1].set_xticks(x_axis)
        ax[1].set_xticklabels(groups_types, rotation=45, ha='right')
        
        ax[0].set_xlabel("Prediction spans.")
        ax[0].set_ylabel("Ground-truth spans.")
        
        plt.tight_layout()
        #plt.show()
        
    
    E_ZM_gt = np.sum(errors_z)/np.sum([gt_s[1]-gt_s[0] for gt_s in gt_spans])
    
    if return_groups:
        return {"ZME": E_ZM_gt}, groups
    else:
        return {
            "ZME": E_ZM_gt
        }