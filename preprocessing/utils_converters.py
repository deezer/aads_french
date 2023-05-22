"""

*** CONVERTERS *** 

<char_spans_to_tok_list_labels>
    --> project spans in terms of characters to labels on words 
<span_to_seq>
    --> append juxtaposed spans into sequences
<span_to_bio>
    --> turn spans indexed on word-tokens to BIO or related schemes
<clean_labels_from_prefixes>
    --> remove prefixes from list of labes (BIO/etc. --> binary)
"""

import pandas as pd 
import numpy as np
import itertools


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%% UTILS CONVERTERS %%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def char_spans_to_tok_list_labels(df:pd.DataFrame
                                  , spans:list
                                  , label_pos = "DS"
                                  , label_neg = "O"
                                  , scheme=None
                                 )->list:
    """
    Transform labels given as spans to list of labels corresponding to tokens.
    
    Arguments:
        df (pd.DataFrame)
            tokenized text with columns "token_idx" (<File.df_tokens>)
        spans (list)
            annotation as spans indexed on text's characters
        label_pos (str, default: "DS")
            positive labels (within the spans from field <char_spans_labels>)
        label_neg (str, default: "O")
            negative labels (outside of the spans from field <char_spans_labels>)
        scheme (str, default: None) --> /!\ supports only "BIO"
            annotation scheme to project labels from characters spans to word-tokens
            - None results in binary labels label_pos|label_neg

    Returns:
        (list) list having the same length as the input df
               mapping each word-token with a label based on the input character spans
    """
    
    # 1. Sort the spans in ascending order (reading direction)
    spans = sorted(spans, key=lambda x: x[0])
    
    # 2. Project <label_pos> on word-tokens that start within any annotation span | otherwise <label_neg>
    labels = [label_pos if np.any([chunk[0] <= tok_id[0] < chunk[1] for chunk in spans])
              else label_neg 
              for tok_id in df["token_idx"]
             ]
    
    # 3. [If <scheme>=="BIO"] --> use BIO conventions: |label_neg|B-label_pos|I-label_pos|I-label_pos|label_neg|
    if scheme=="BIO":
        # 3.1 Get indices of the beginning of the annotated chunks
        b_labels_indx = [chunk[0] for chunk in spans]
        
        curr_span_id=0
        nb_spans=len(b_labels_indx)
        
        # 3.2 Iterate over the tokens 
        for i, (lab, tok_id) in  enumerate(zip(labels, list(df["token_idx"]))):
            # 3.3 If the token is in a span (annotated as label_pos)
            if (lab==label_pos):
                # 3.3.1 --> add prefix "I-"
                bio_lab = "I-{lab}".format(lab=lab)
                
                # 3.3.2 If the token is the first positive one after the currently
                #.      considered annotated span ==> add prefix "B-", ie. it is the start of this span
                if curr_span_id<nb_spans:
                    if (tok_id[0]>=b_labels_indx[curr_span_id]):
                        bio_lab = "B-{lab}".format(lab=lab)
                        curr_span_id += 1
                        
                # 3.4 replace binary label with BIO label
                labels[i] = bio_lab
        
    
    return labels

def span_to_seq(tok_spans:list
                , tok_list:list
                , from_boolean_list:bool=False
               )->list:  
    """
    Megre juxtaposed spans into sequences
    
    /!\ To be comparable to models predictions (made at the token level which cannot produce inner boundaries),
        the spans are converted into sequences, so that if 2 spans follow each other,
        then they are considered as 1 sequence.
        ie. O B-DS E-DS B-DS E-DS O
         -> O B-DS I-DS I-DS E-DS O
    
    Arguments:
        tok_spans (list)
            Spans (supposedly indexed upon tokens)
        tok_list (list)
            list of tokens (to get the total length of the text and have comparable output for different annotations
            — must to correspond to <tok_spans> 
        from_boolean_list (bool:False)
            given list of bools (w/ True on positive labels indices)
    Returns:
        (list): list of sequences, ie. merged spans
        
    Example:
    span_to_seq(tok_spans=[[0,3],[3,7]]
            , tok_list=["-", "Bonjour", ".", "-", "Bonjour", ".", "répondit", "l'","autre","."]
           )
    >>> [(0, 7)]
    """
    
    if from_boolean_list:
        boolean_labels = tok_list
    else:
        boolean_labels = [np.any([chunk[0] <= tok_id < chunk[1] for chunk in tok_spans])
                          for tok_id, tok in enumerate(tok_list)
                         ]
    
    tok_seqs = [(g[0][0], g[0][0]+len(g))
                for i, (key, group) in enumerate(itertools.groupby(enumerate(boolean_labels), key=lambda v: v[1]))
                if key
                for g in (list(group),)
               ]
    
    return tok_seqs

def span_to_bio(tok_spans:list
                , tok_list:list
                , scheme:str = None#"IOBES" # IOB1, IOB2, IOE1, IOE2, IOBES, BILOU
                , from_boolean_list:bool=False
               )->list:
    """
    From list of spans to corresponding list of labels with choosen annotation scheme
    
    /!\ To be comparable to models predictions (made at the token level which cannot produce inner boundaries),
        the spans are first converted into sequences, so that if 2 spans follow each other,
        then they are considered as 1 sequence.
        ie. O B-DS E-DS B-DS E-DS O
         -> O B-DS I-DS I-DS E-DS O
    
    Arguments:
        tok_spans (list)
            Spans (supposedly indexed upon tokens)
        tok_list (list)
            list of tokens (to get the total length of the text and have comparable output for different annotations
            — must to correspond to <tok_spans> 
        scheme (str: IOB2, IOE2, IOBES, None)
            Convention for the annotation:
             `None`: O |  A  |  A  |  A  | O |  A  |  A  | O |  A  | O
               IOB2: O | B-A | I-A | B-A | O | B-A | I-A | O | B-A | O
               IOE2: O | I-A | E-A | I-A | O | I-A | E-A | O | E-A | O
              IOBES: O | B-A | E-A | S-A | O | B-A | E-A | O | S-A | O
    Returns:
        (list): list of annotations with the chosen convention
        
    Example:
    span_to_bio(tok_spans = [[3,6], [6,9]],
                 tok_list = ["Il", "dit",":","-","Bonjour","!", "-", "Bonjour" ".", "répondit", "l'", "autre", "." ],
                 scheme="IOBES"
                )
    >>> ['O', 'O', 'O', 'B-DS', 'I-DS', 'I-DS', 'I-DS', 'I-DS', 'E-DS', 'O', 'O', 'O']
    """
    possible_schemes = ["IOB2", "IOE2", "IOBES", None]
    assert scheme in possible_schemes, "scheme must be one of : {possible}. Input {i} is not.".format(possible=possible_schemes,i=scheme)
    
    # First get a list of binary labels, then annotate with the corresponding scheme
    # as the predictions formalization (token level) cannot produce inner booundaries
    tok_seqs = span_to_seq(tok_spans=tok_spans
                           , tok_list=tok_list
                           , from_boolean_list=from_boolean_list
                          )
    
    if not scheme:
        bio_labels = ["DS" if np.any([chunk[0] <= tok_id < chunk[1] for chunk in tok_seqs])
                      else "O" 
                      for tok_id, tok in enumerate(tok_list)
                     ]
    else:
        bio_labels = ["I-DS" if np.any([chunk[0] <= tok_id < chunk[1] for chunk in tok_seqs])
                      else "O" 
                      for tok_id, tok in enumerate(tok_list)
                     ]
        
        if scheme=="IOBES":
            for chunk in tok_seqs:
                if chunk[1]-chunk[0]<2:
                    bio_labels[chunk[0]]="S-DS"
                else:
                    bio_labels[chunk[1]-1]="E-DS"
                    bio_labels[chunk[0]]="B-DS"
                    
        elif scheme=="IOB2":
            for chunk in tok_seqs:
                bio_labels[chunk[0]]="B-DS"
                
        elif scheme=="IOE2":
            for chunk in tok_seqs:
                bio_labels[chunk[1]-1]="E-DS"
        
    
    return bio_labels

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%% Label cleaner %%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

_POSSIBLE_PREFIXES = ["B-", "I-", "E-", "S-"]

def clean_labels_from_prefixes(labels_list:list=[],
                               used_labels:list=[],
                               prefixes_list:list=_POSSIBLE_PREFIXES,
                              )->list:
    """
    Turn BIO/BIOES/etc. annotations schemes into binary IO scheme 
    
    Arguments:
        labels_list (list, default:[])
            list of labels to be cleaned from prefixes (turn to binary labels)
        used_labels (list, default:[])
            list of labels used
        prefixes_list (list, default:["B-", "I-", "E-", "S-"])
            list of prefixes to be removed
        
    Returns:
        (list) the input <labels_list> with prefixes removed
    """
    
    if len(used_labels)==0:
        used_labels = set(labels_list)
    
    if np.any([p in lab for p in prefixes_list for lab in used_labels]):
        # SANITY CHECK: Annotation must be coherent (no mixing between binary and BIO)
        if not np.all([lab=="O" or "-" in lab for lab in labels_list]):
            raise ValueError("Incorrect encoding scheme")
        labels_list = [lab[2:] if lab!="O" else lab for lab in labels_list]
    
    return labels_list