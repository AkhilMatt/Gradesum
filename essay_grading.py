#Standard
import re
import pickle

#Third-party
import pandas as pd
import nltk
import numpy as np
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

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




clean_essays = pd.read_csv('clean_essays.csv')
def grade(essay):
        essay_df = pd.DataFrame({'essay':preprocess(essay), 'clean_essay':lem(essay)}, index = [0])
        essay_df['Char_count'], essay_df['Word_count'], essay_df['Avg_word_len'], essay_df['Unique_words'] = counts(essay)
        essay_df['noun_count'], essay_df['adj_count'], essay_df['verb_count'], essay_df['adv_count'] = count_pos(essay_df['essay'][0])
        SentenceCount = 0
        for char in essay_df['essay'][0]:
            if char == '.':
                SentenceCount += 1
        essay_df['Sentence_count'] = SentenceCount
        vectorizer = CountVectorizer(max_features = 10000, ngram_range=(1, 3), stop_words='english')
        # Use preprocessed essays to make count vectors
        count_vectorfit = vectorizer.fit(clean_essays['clean_essay'])
        count_transform = vectorizer.transform(essay_df['clean_essay'])
        feature_names = vectorizer.get_feature_names()
        X = count_transform.toarray()
        feature_df = essay_df[['Char_count', 'Word_count', 'Avg_word_len', 'Unique_words', 'Sentence_count', 'noun_count', 'adj_count', 'verb_count', 'adv_count']]
        X_test = np.concatenate((feature_df, pd.DataFrame(X)), axis = 1)
        model=pickle.load(open("model\\svr_model",'rb'))
        ypred=round(float(model.predict(X_test)),4)
        return ypred
