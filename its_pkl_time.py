# This is the file I use to create my pkl file.
# I use this so I can just load up my already trained an prepaired model so I don't have to do it every time.

# IT'S FREAKING PICKLE TIME!
print 'Training my model...'
print 'PICKLE TIME!!!!!!!!!'

# Importing my standard stuff
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
# Setting up my pickling skillz, yo
from sklearn.externals import joblib

# Here I combined my training & test data. I will used this as main total "training" data in production & for cross_val stuff.
corpus = pd.read_csv('data/total_data.csv')
# Getting our list of "badwords"
badwords = set(pd.read_csv('data/my_badwords.csv').words)


# Setting Up my X & y
X_train = corpus.Comment
y_train = corpus.Insult

# Here I added addtional features

# This is just a count of how many bad words
train_badwords_count = []
test_badwords_count = []

for el in X_train:
    tokens = el.split(' ')
    train_badwords_count.append(len([i for i in tokens if i.lower() in badwords]))

# **SHOUT OUT**
# I was messing with stuff from Andreas Mueller for these next features. Thanks man! :)
# His Blog Post on this: http://blog.kaggle.com/2012/09/26/impermium-andreas-blog/
# **SHOUT OUT**
train_n_words = [len(c.split()) for c in X_train]
train_allcaps = [np.sum([w.isupper() for w in comment.split()]) for comment in X_train]
train_allcaps_ratio = np.array(train_allcaps) / np.array(train_n_words, dtype=np.float)
train_bad_ratio = np.array(train_badwords_count) / np.array(train_n_words, dtype=np.float)
train_exclamation = [c.count("!") for c in X_train]
train_addressing = [c.count("@") for c in X_train]
train_spaces = [c.count(" ") for c in X_train]

# Setting up or vect and clf instances
vect = CountVectorizer(binary=True)
clf = LogisticRegression()

# Fitting and transforming my X_Train using Count Vectorizer
X_train_transform = vect.fit_transform(X_train)

# Reshaping my new features so I can stick them in my X_train
train_reshaped_badwords = np.array(train_badwords_count).reshape((len(train_badwords_count),1))
train_reshaped_n_words = np.array(train_n_words).reshape((len(train_badwords_count),1))
train_reshaped_allcaps = np.array(train_allcaps).reshape((len(train_badwords_count),1))
train_reshaped_allcaps_ratio = np.array(train_allcaps_ratio).reshape((len(train_badwords_count),1))
train_reshaped_bad_ratio = np.array(train_bad_ratio).reshape((len(train_badwords_count),1))
train_reshaped_exclamation = np.array(train_exclamation).reshape((len(train_badwords_count),1))
train_reshaped_addressing = np.array(train_addressing).reshape((len(train_badwords_count),1))
train_reshaped_spaces = np.array(train_spaces).reshape((len(train_badwords_count),1))

# Adding my new features into my X_train
X_train_transform = np.hstack((X_train_transform.todense(), train_reshaped_badwords))
X_train_transform = np.hstack((X_train_transform, train_reshaped_n_words))
X_train_transform = np.hstack((X_train_transform, train_reshaped_allcaps))
X_train_transform = np.hstack((X_train_transform, train_reshaped_allcaps_ratio))
X_train_transform = np.hstack((X_train_transform, train_reshaped_bad_ratio))
X_train_transform = np.hstack((X_train_transform, train_reshaped_exclamation))
X_train_transform = np.hstack((X_train_transform, train_reshaped_addressing))
X_train_transform = np.hstack((X_train_transform, train_reshaped_spaces))

# Fitting our Logisitc Regression Classifier
clf.fit(X_train_transform,y_train)



# Actually creating my pickle files for later use. One for my classifier and one for my vectorizerT
joblib.dump(clf, 'clf.pkl')
joblib.dump(vect, 'vect.pkl')
