"""

*** Rule-based Regular Expression Quote Detection Module ***

This script contains 2 RegEx baselines:
    1. The first one inspired from BookNLP approach
        Idea: 
            - Find most common DS marker convention used in a file
            - Define the corresponding RegEx
            - Identify matching passage
            
    2. The second one copied from Byszuk (2020) baseline
        Idea:
            - Compile list of each possible DS marker
            - Define RegEx for each of these DS markers
            - Identify ALL matching patterns
            
--> The two RegEx baselines mainly differ in the first step:
    (1 - BookNLP) assumes that (a) only one DS convention per file is used
                      AND that (b) it is the most common one in the text, while
    (2 - Byszuk) does not make any of these hypothesis and identifies any pattern
                    that could be DS as DS

"""

from preprocessing.data_utils import File, CorpusDict

from collections import Counter

import re

### List of symbols that can be used to mark DS in French literature
## Dashes
# unicode id
dashes_list = [45,
               6150,
               8210, 
               8211,
               8212,
               8213
              ]
# convert unicode id to glyphs
dashes_glyphs = [chr(dash_c) for dash_c in dashes_list]
dashes_glyphs += [dash+dash for dash in dashes_glyphs] # include double dashes s.a. in -- Bonjour ! dit-il

## Quotation marks
# unicode id
marks_list_left = [171
                   , 10077
                   , 8220
                   , 12317
                  ]
# convert unicode id to glyphs
marks_glyphs_left = [chr(mark_l) for mark_l in marks_list_left]
# same for right markers: “[”]
marks_list_right = [187
                    , 10078
                    , 8221
                    , 12318
                   ]
marks_glyphs_right = [chr(mark_r) for mark_r in marks_list_right]                 

# Store into dict + compute corresponding regex
possible_quote_markers_dicts = {}
for dash in dashes_glyphs:
    curr_dict = {}
    curr_dict["symbol"] = dash
    curr_dict["type"] = "dash"
    curr_dict["right"] = "\n"
    # -> (--.+?(?=--|\n))
    curr_dict["regex_catch"] = r"("+dash+".+?(?="+dash+"|\n))"
    
    possible_quote_markers_dicts[dash]=curr_dict

for left_m, right_m in zip(marks_glyphs_left, marks_glyphs_right):
    curr_dict = {}
    curr_dict["symbol"] = left_m
    curr_dict["type"] = "markers"
    curr_dict["right"] = [right_m, "\n"]
    # («[^»\n]*(»|\n))
    # -> («.+(?:(«.+?» ))?(»|\n)) => to handle inner quotes + more coherent with other
    curr_dict["regex_catch"] = r"("+left_m+"[^"+right_m+"\n]*("+right_m+"|\n))"
    
    possible_quote_markers_dicts[left_m]=curr_dict

#possible_quote_markers = [k for k in possible_quote_markers_dicts.keys()]


def get_most_common_quote_marker(tokens:list
                                 , quote_markers:dict = possible_quote_markers_dicts
                                 , prop_th = 0.
                                )->dict:
    """
    Arguments:
        tokens (list of strings)
            List of tokens from the analyzed text
        quote_markers (dict = possible_quote_markers_dicts)
            Dictionnary with keys being possible quotation markers 
            and values being additional information for each of these 
            (as dicts {'symbol', 'right', 'type', 'regex_catch'})
    Returns:
        (dict): corresponding to values stored in `quote_markers` under the key being the most used among `tokens`
        
    Example:
    get_most_common_quote_marker(['--', 'Oui', ',', '«', 'pronto', '»' '!', '--', 'Allo', '!'])
    >>> {'symbol': '--', 'type': 'dash', 'right': '\n', 'regex_catch': '(--.+?(?=--|\n))'}
    """

    quote_symbol = "—" # default value
    quote_symbols=Counter()

    for i, tok in enumerate(tokens):
        for marker in quote_markers.keys():
            if tok==marker:
                quote_symbols[marker]+=1
                
    
    # ASSUMPTION: quote are marked with most common symbol
    if len(quote_symbols) > 0:
        most_commons = quote_symbols.most_common()
        quote_symbol=most_commons[0][0]
        
        # HEURISTICALLY: try to avoid simple "-", also used in words eg. "celui-là", or junctions s.a. "dit-il"
        if (quote_symbol=="-") and len(most_commons)>1:
            if (most_commons[1][1]>prop_th*most_commons[0][1]):
                quote_symbol = most_commons[1][0]
                    
    return quote_markers[quote_symbol]

def get_regex_quotes(txt:str
                     , regex_rule:str#=regex_tiret+'|'+regex_quotation_mark
                    )->list:
    """
    Returns list of indexes (in characters) of spans (start, end) of text segments matching Regex rule.
    
    Example:
    get_regex_quotes('Pour illustrer, il dit: «Voici un exemple».')
    >>> [(24, 42)]
    """
    quote_spans_regex = [(m.start(0), m.end(0)) for m in re.finditer(regex_rule, txt)]
    
    return quote_spans_regex

def bookNLP_like_quotes(corpus_file:File
                        ,prop_th=0.
                       )->list:
    """
    Retrieve most used symbol from a predefined list in the tokenized text.
    Assume it is then the marker for Direct Speech.
    Extract quotes accordingly using regular expression.
    """
    file_tokens = list(corpus_file.df_tokens["token"])
    file_text = corpus_file.text
    
    quote_symbol_used_dict = get_most_common_quote_marker(tokens=list(file_tokens)
                                                          , prop_th=prop_th
                                                         )
    
    bookNLP_regex = quote_symbol_used_dict["regex_catch"]
        
    bookNLP_quotes = get_regex_quotes(txt=file_text
                                      , regex_rule=bookNLP_regex
                                     )
    
    return bookNLP_quotes


# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
def byszuk_baseline(corpus_file:File
                   )->list:
    """
    Copied and adapted from byszuk_2020 baseline implementation based on regex
    """
    
    raw_text = corpus_file.text
    
    patterns_quotes = [
        re.compile(r"(\".+?\")"),
        re.compile(r"(“.+?”)"),
        re.compile(r"(„.+?”)"),
        re.compile(r"(„.+?”)"),
        re.compile(r"(».+?«)"),
        re.compile(r"(«.+?»)"),
        re.compile(r"(›.+?‹)"),
        re.compile(r"(›{2}.+?‹{2})"),
        re.compile(r"(„.+?“)"),
        re.compile(r"(\'{2}.+?\'{2})"),
        re.compile(r"(‘{2}.+?‘{2})"),
        re.compile(r"(’{2}.+?’{2})"),
        re.compile(r"(‘.+?’)"),
        re.compile(r"(‘{2}.+?’{2})"),
        re.compile(r"(❝.+?❞)"),
        re.compile(r"(❞.+?❝)"),
        re.compile(r"(〝.+?〞)"),
        re.compile(r"(〞.+?〝)")       
    ]
    pattern_dashes = re.compile(r"^\s*([-֊᠆‐‑⁃﹣－‒–—⁓╌╍⸺⸻⹃〜〰﹘]{1,2}.+?)$")
    pattern_split = re.compile(r"(\s{1}[-֊᠆‐‑⁃﹣－‒–—⁓╌╍⸺⸻⹃〜〰﹘]{1,2}\s{1})|([-֊᠆‐‑⁃﹣－‒–—⁓╌╍⸺⸻⹃〜〰﹘]{1,2}\s{1})|(\s{1}[-֊᠆‐‑⁃﹣－‒–—⁓╌╍⸺⸻⹃〜〰﹘]{1,2})")
    
    
    # In Byszuk: iterate over paragraphs and try match
    # Here: simulate paragraphs using breaklines ?
    paragraphs = [(m.group(0), (m.start(), m.end()-1)) for m in re.finditer(r'(.+?)\n', raw_text)]
    if len(paragraphs)==0:
        paragraphs = [(raw_text, (0, len(raw_text)-1))]
        
    # In Byszuk: write in a new file
    # Here: directly return spans
    #new_content = '<?xml version="1.0" encoding="UTF-8"?>\n'
    quotes_spans_byszuk = []

    for p_content, (p_start, p_end) in paragraphs:

        match_dashes = re.match(pattern_dashes, p_content)
        matches_quotes = [bool(re.search(pattern, p_content)) for pattern in patterns_quotes]

        saids = []
        if match_dashes != None:
            p_dialog = match_dashes.group()
            p_dialog_splitted = [i for i in re.split(pattern_split, p_dialog) if i != '' and i != None]
            if len(p_dialog_splitted) < 2:
                saids = p_dialog_splitted
            elif len(p_dialog_splitted) == 2:
                saids = ["".join(p_dialog_splitted)]

            else:
                for i in range(1, len(p_dialog_splitted), 4):
                    try:
                        saids.append("".join([
                            p_dialog_splitted[i-1],
                            p_dialog_splitted[i],
                            p_dialog_splitted[i+1]
                        ]).strip())
                    except IndexError:
                        saids.append("".join([
                        p_dialog_splitted[i-1],
                        p_dialog_splitted[i]
                        ]).strip())

        elif any(matches_quotes):
            if '»' in p_content and '«' in p_content and p_content.index('»') < p_content.index('«'):
                pattern = patterns_quotes[4]
            elif '»' in p_content and '«' in p_content and p_content.index('»') > p_content.index('«'):
                pattern = patterns_quotes[5]
            else:
            # get first matching pattern
                pattern = patterns_quotes[matches_quotes.index(True)]
            for m in re.finditer(pattern, p_content):
                saids.append(m.group())

        else:
            # write in new file 
            #new_content += p_content + '</p>\n'

            continue

        match_hits = []
        for said in saids:
            match = re.search(re.escape(said), p_content)
            match_hits.append([match.start(), match.end()])


        pos = 0
        for i, (start, end) in enumerate(match_hits, 1):
            if start == 0 and pos == 0 and end < len(p_content) - 1:
                # write in new file 
                #new_content += '<said>' + p_content[start:end] + '</said>'
                quotes_spans_byszuk += [(p_start+start, p_start+end)]
                pos = end 

            else:
                # write in new file 
                #new_content += p_content[pos:start]
                #new_content += '<said>' + p_content[start:end] + '</said>'
                quotes_spans_byszuk += [(p_start+start, p_start+end)]
                pos = end

            # no DS
            #if i == len(match_hits) and len(p_content) > end:
                #new_content += p_content[end:]
                
    return quotes_spans_byszuk