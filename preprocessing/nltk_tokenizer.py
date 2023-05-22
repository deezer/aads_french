"""

*** Define Document maker with NLTK regex tokenization ***

The tokenizer pipeline should return a document with field 
    <sents> that contains the list of sentences and each sentence should be represented as
    a list of Token objects with fields <text> and <idx>

--> This makes the defined tokenizer pipeline usable as SpaCy pipeline in our framework

"""

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%% NLTK Tokenizers %%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import RegexpTokenizer
import nltk.tokenize.punkt as pt

class CustomLanguageVars(pt.PunktLanguageVars): # Modified to keep break lines in sentence tokenization

    _period_context_fmt = r"""
        \S*                        
        %(SentEndChars)s           
        \s*                     
        (?=(?P<after_tok>
            %(NonWord)s            
            |
            (?P<next_tok>\S+)   
        ))"""

sentence_tokenizer = pt.PunktSentenceTokenizer(lang_vars=CustomLanguageVars())

regexp_tokenizer = RegexpTokenizer(r'''\w'|\w+|[^\w\t ]''')

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%% Text Classes %%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

class Document:
    """
    TODO:
    - [ ] DOCUMENTATION 
    """
    def __init__(self,
                 text:str,
                 sentences_list,
                ):
        self.text = text
        self.sents = sentences_list
        
        
class Token:
    """
    TODO:
    - [ ] DOCUMENTATION 
    """
    def __init__(self,
                 char_index,
                 word,
                ):
        self.idx = char_index
        self.text = word
        

        
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%% Tokenizer pipeline %%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
def nltk_tokenizer_pipe(text:str)->Document:

    """

    Make NLTK tokenizer comparable to SpaCy tokenizers
    --> return Objects with used features:
        For document:
            - sents
                -> list of word-tokens
        For words:
            - idx
            - text
            
    Arguments:
        text (str)
            Text to be tokenized

    Returns:
        (Document) Tokenized text: document containing a list of sentences in field <sents>
                    each of them containing a list of Token objects with fields <text> and <idx>

    """
    # 0. Instantiate empty list of sentences to be filled
    doc_sents = []
    
    # 1. For each span of text recognized as a sentence
    for sent_span in sentence_tokenizer.span_tokenize(text):
        # 1.1 Retrieve the corresponding string in text
        sent_text = text[sent_span[0]:sent_span[1]]
        
        sent_tokens = []
        # 1.2 Tokenize the retrieved sentence
        for tok_span in regexp_tokenizer.span_tokenize(sent_text):
            sent_tokens += [Token(char_index=sent_span[0]+tok_span[0],
                                  word=sent_text[tok_span[0]:tok_span[1]]
                                 )
                           ]
        doc_sents += [sent_tokens]
        
    # 2. Make a Document object with the sentences
    document = Document(text=text,
                        sentences_list=doc_sents
                       )
                        
    return document