#Standard
import re

#Third-party
import pandas as pd
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

#Local



def preprocess(essay, remove_stopwords = True):
    '''
    Remove stopwords and special characters.
    Return preprocessed essay as string
    '''
    essay = re.sub("[^a-zA-Z0-9]", " ", essay)
    words = essay.lower().split()
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    words = ' '.join(words).lower()
    return words

def get_wordnet_pos(word):
    '''
    Getting the right POS tag for lemmatizer
    '''
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def lem(essay):
    '''
    Returns lemmatized essay
    '''
    lemmatizer = WordNetLemmatizer()
    essay = [lemmatizer.lemmatize(w, pos = get_wordnet_pos(w)) for w in essay.split()]
    essay = ' '.join(essay)
    return essay

def counts(x):
    '''
    Returns average word length and counts of characters, words and unique words 
    '''
    CharCount = len(x)
    WordCount = len(x.split())
    W = [len(w) for w in x.split()]
    AvgWordLen = sum(W)/len(W)
    AvgWordLen = round(AvgWordLen, 4)
    UniqueWords = len(set(x.split()))
    return(CharCount, WordCount, AvgWordLen, UniqueWords)


def count_pos(x):
    '''
    Part of speech tagger. Unknown words tagged as noun, if it doesn't end in 'ing'
    '''
    noun_count = 0
    adj_count = 0
    verb_count = 0
    adverb_count = 0
    for pos in nltk.pos_tag(x.split()):
        tag = pos[1]
        if tag[0] == 'N':
            noun_count += 1
        elif tag[0] == 'V':
            verb_count += 1
        elif tag[0] == 'J':
            adj_count += 1
        elif tag[0] == 'R':
            adverb_count += 1
    return noun_count, verb_count, adj_count, adverb_count


