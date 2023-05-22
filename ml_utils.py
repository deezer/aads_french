"""
*** Machine Learning utils module ***

<create_sentences_chunks>
    --> segments tokenized text into chunks shorter than a given length (in terms of tokens)
"""

import pandas as pd
import ast
import logging

def create_sentences_chunks(data:pd.DataFrame
                            , maxlen:int=128
                            , compare_column:str="label"
                            , eof_tag:str = "EOF"
                           )->list:
    
    """ TAKEN AND ADAPTED FROM REDEWIEDERGEBE (BRUNNER, 2020)
    
    Takes a pandas dataframe with columns 'token', 'token_idx' and 'sentstart' and creates a list of chunks.
    
    Each chunk may contain several real sentences, but at most <maxlen> tokens.
    Thus, the individual chunks are often shorter than <maxlen>.
    
    Sentences are appended to chunks in the reading directions (without optimization).
    Only sentences longer than <maxlen> can be splitted among several chunks.
    
    No file boundaries are crossed. Using "EOF" tags distinguishing rows associated with one file from another,
    each individual chunk can not contain excerpts from more than one file.

    Arguments:
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
        eof_tag (str, default: "EOF")
            special token separating one file from another in <data>
    Returns:
        (list) List of text chunks no longer than <maxlen>, preserving sentences integrity (while shorter than <maxlen>),
                as well as files integrities.
                Each chunk is in fact a list of lists [tokens, tokens_start_ids, labels]
                                                       |            |             |
                                           List of tokens      Corresponding      Corresponding
                                         making the chunk      tokens' start             labels
                                                               indices
    """
    
    chunks_list = [] # [[toks, tokstarts, tags], [toks, tokstarts, tags], ...]
    
    tokstartlist = []
    toklist = []
    taglist = []
    
    # check if the <compare_column> labelled columned is in the DataFrame:
    has_labels = compare_column in data.columns
    
    # track the sentence that is currently being processed
    curr_sentence_tok = []
    curr_sentence_tag = []
    curr_sentence_starts = []

    for index, row in data.iterrows():
        
        tok = str(row["token"])
        
        if compare_column != "NaN":
            tag = str(row[compare_column])
        else:
            tag = "O"
            
        # if the current token is "EOF" this marks the end of sample file
        # chunks may not cross file boundaries, therefore end the sentence here in any case
        if tok == eof_tag:
            # do not add this token to any list
            # merge toklist and curr_sentence_tok list to get all current tokens
            toklist.extend(curr_sentence_tok)
            taglist.extend(curr_sentence_tag)
            tokstartlist.extend(curr_sentence_starts)

            #     
            if len(toklist) > 0:
                chunks_list.append({"tokens":toklist,
                                     "tokens_ids":tokstartlist,
                                     "tags":taglist
                                    })

            toklist = []
            taglist = []
            tokstartlist = []
            # reset the curr sent lists as well
            curr_sentence_tok = []
            curr_sentence_tag = []
            curr_sentence_starts=[]

        else:
            if type(row["token_idx"])==str:
                tokstart = int(ast.literal_eval(row["token_idx"])[0])
            else:
                tokstart = row["token_idx"][0]
            # if we are at the start of a new sentence, add the contents of curr_sentence_tok
            # and curr_sentence_cat to the main lists and start a new curr_sentence
            if row["sentstart"] == "yes":
                # Add previous sentence to the chunk
                toklist.extend(curr_sentence_tok)
                taglist.extend(curr_sentence_tag)
                tokstartlist.extend(curr_sentence_starts)
                # Start filling new sentence
                curr_sentence_tok = [tok]
                curr_sentence_tag = [tag]
                curr_sentence_starts = [tokstart]
            else:
                # Continue filling current sentence
                curr_sentence_tok.append(tok)
                curr_sentence_tag.append(tag)
                curr_sentence_starts.append(tokstart)

            # if the combined length of toklist and curr_sentence_tok is > maxlen now,
            # create a flair sentence with the tokens in toklist and reset it
            # the remaining tokens in curr_sentence_tok are saved for the next chunk
            if len(toklist) + len(curr_sentence_tok) > maxlen:
                # if toklist is empty at this point, we have a sentence > maxlen
                # and must split it. The last token currently in curr_sentence will
                # be preserved for later so that the chunk is not too long
                if len(toklist) == 0:
                    toklist.extend(curr_sentence_tok[0:-1])
                    taglist.extend(curr_sentence_tag[0:-1])
                    tokstartlist.extend(curr_sentence_starts[0:-1])

                    curr_sentence_tok = [curr_sentence_tok[-1]]
                    curr_sentence_tag = [curr_sentence_tag[-1]]
                    curr_sentence_starts = [curr_sentence_starts[-1]]

                # Store current chunk
                if len(toklist)>0:
                    chunks_list.append({"tokens":toklist,
                                         "tokens_ids":tokstartlist,
                                         "tags":taglist
                                        })

                toklist = []
                taglist = []
                tokstartlist = []
                
    # if the loop is complete, empty the buffers and add them to the list
    # For security as it should not be the case with EOF at the end of last file
    if len(curr_sentence_tok) > 0 or len(toklist) > 0:
        toklist.extend(curr_sentence_tok)
        taglist.extend(curr_sentence_tag)
        tokstartlist.extend(curr_sentence_starts)
        
        # Store chunk
        if len(toklist)>0:
            chunks_list.append({"tokens":toklist,
                                 "tokens_ids":tokstartlist,
                                 "tags":taglist
                                })
            
    return chunks_list