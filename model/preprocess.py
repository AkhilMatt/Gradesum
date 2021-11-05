#Standard

#Third-party
import pandas as pd
import numpy as np
from sklearn.preprocessing import minmax_scale

#Local 
from data_preprocessing import preprocess, lem, counts, count_pos

data = pd.read_csv('data\\training_set_rel3.tsv', sep = '\t', encoding = 'ISO-8859-1')
data = data.dropna(axis = 1)
data = data.drop(['rater1_domain1', 'rater2_domain1'], axis = 1)

set1 = data[data['essay_set'] == 1].reset_index().drop('index', axis = 1)
set2 = data[data['essay_set'] == 2].reset_index().drop('index', axis = 1)
set3 = data[data['essay_set'] == 3].reset_index().drop('index', axis = 1)
set4 = data[data['essay_set'] == 4].reset_index().drop('index', axis = 1)
set5 = data[data['essay_set'] == 5].reset_index().drop('index', axis = 1)
set6 = data[data['essay_set'] == 6].reset_index().drop('index', axis = 1)
set7 = data[data['essay_set'] == 7].reset_index().drop('index', axis = 1)
set8 = data[data['essay_set'] == 8].reset_index().drop('index', axis = 1)
# Scaling scores of essays with respect to the essay set
set1['new_score']= minmax_scale(set1['domain1_score'])*10
set2['new_score']= minmax_scale(set2['domain1_score'])*10
set3['new_score']= minmax_scale(set3['domain1_score'])*10
set4['new_score']= minmax_scale(set4['domain1_score'])*10
set5['new_score']= minmax_scale(set5['domain1_score'])*10
set6['new_score']= minmax_scale(set6['domain1_score'])*10
set7['new_score']= minmax_scale(set7['domain1_score'])*10
set8['new_score']= minmax_scale(set8['domain1_score'])*10

new_data = pd.concat([set1,set2,set3,set4,set5,set6,set7,set8], axis='rows')
new_data = new_data.reset_index().drop('index', axis = 1)
# Stop word and puntuation removal 
new_data['clean_essay'] = new_data['essay'].apply(lambda x: preprocess(x))

# Lematization.
new_data['clean_essay'] = new_data['clean_essay'].apply(lambda x:lem(x))

# Count features.
for index, essay in enumerate(new_data.essay):
    Char_count, Word_count, Avg_word_size, Unique_words = counts(essay)
    new_data.loc[index, 'Char_count'] = Char_count
    new_data.loc[index, 'Word_count'] = Word_count
    new_data.loc[index, 'Avg_word_len'] = Avg_word_size
    new_data.loc[index, 'Unique_words'] = Unique_words
    SentenceCount = 0
    for char in essay:
        if char == '.':
            SentenceCount += 1
    new_data.loc[index, 'Sentence_count'] = SentenceCount
 

# Pos tag counts
new_data['noun_count'], new_data['adj_count'], new_data['verb_count'], new_data['adv_count'] = zip(*new_data['essay'].map(count_pos))

##new_data['POS'] = pd.Series(map(count_pos, new_data.essay))
##for index, tup in enumerate(new_data['POS']):  
##    new_data.loc[index, 'noun_count'] = tup[0]
##    new_data.loc[index, 'verb_count'] = tup[1]
##    new_data.loc[index, 'adj_count'] = tup[2]
##    new_data.loc[index, 'adverb_count'] = tup[3]
##new_data.drop('POS', axis = 1, inplace = True)



new_data.to_csv('new_data.csv', index = False)


