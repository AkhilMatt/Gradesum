#Standard
import pickle

#Third-party
import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error



df = pd.read_csv('new_data.csv')

# Count vectors of essay
vectorizer = CountVectorizer(max_features = 10000, ngram_range=(1, 3), stop_words='english')
count_vectors = vectorizer.fit_transform(df['clean_essay'])
feature_names = vectorizer.get_feature_names()
X = count_vectors.toarray()
feature_df = df[['Char_count', 'Word_count', 'Avg_word_len', 'Unique_words', 'Sentence_count', 'noun_count', 'adj_count', 'verb_count', 'adv_count']]

X_full = np.concatenate((feature_df, pd.DataFrame(X)), axis = 1)
y_full = df['new_score']
X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size = 0.3, random_state = 0)

# Chosen parameters after hyper-parameter tuning
svr = SVR(C=1.0, epsilon=0.2)
svr.fit(X_train, y_train)
pickle.dump(svr, open("svr_model",'wb'))

model = pickle.load(open("svr_model",'rb'))
y_pred = model.predict(X_test)
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))

