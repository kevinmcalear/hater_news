
# coding: utf-8

# ![](http://www.blogcdn.com/www.urlesque.com/media/2010/05/haterafrican.jpg)
# 
# # Hater News
# 
# Haterz gonna hate. But now you know who the haterz are.
# 
# ###My Goal:
# 
# Be able to look at any user from [hacker news](https://news.ycombinator.com/) and be able to tell if they are a "hater" or not based on a score I give them. 
# 
# ###Feel free to look under the hood.
# 
# [The Classifier Data I'm Using.](https://www.kaggle.com/c/detecting-insults-in-social-commentary/data)
# 
# [Hacker News API I'm Using.](https://github.com/HackerNews/API)
# 
# *Eventually I would love to turn this into a [Chrome App](https://developer.chrome.com/extensions/getstarted) that will just real-time analize any user on a page when you visit hacker news and put a score right next to them so the world can see if they hate or love.*

# #Lets Get Started! 
# We will begin by importing all of the required items we need from things like [scikit learn](http://scikit-learn.org/stable/), [pandas](http://pandas.pydata.org/pandas-docs/stable/10min.html), [numpy](http://www.numpy.org/), and even play around with things like [nlk](http://www.nltk.org/) and [re](https://docs.python.org/2/library/re.html).
# 
# The basic idea will be to split out each comment, create features based on things like if a word shows up in a comment or not, if insulting words appear, and other types of features. Once I have that then I will add in a classifier and if it scores well we are on our way to finding haters!

# In[227]:

# Importing my standard stuff
import pandas as pd
import numpy as np

# Importing different Vectorizers to see what makes my soul sing the loudest.
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# Importing various Classifiers to see what wins out. There can only be one! 
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# Snagging the validation stuff to see how well things do.
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import cross_val_score


# Getting Stopwords to potentially improve accuracy.
from nltk.corpus import stopwords

# Messing with tokenization via NLTK
from nltk import word_tokenize, pos_tag
# Messing with Stemming
from nltk.stem import PorterStemmer
# Messing with Lemmatizing
from nltk.stem.wordnet import WordNetLemmatizer

# Messing with some other string splitting techniques. You can use the following thing like this:
# re.split('; |, ',str)
import re


# #Exploring Different Ways To Achieve My Goal.
# 
# Here I am importing my data using **pandas**, pulling out the right features and assigning them to my X & y to put into my models. You'll notice that I'm actually using both my test & training data in a single variable because of how [cross_val_score](http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.cross_val_score.html) works later on.

# In[228]:

# Setting up my english stopwords, yo.
stopwords = stopwords.words('english')

# Here I combined my training & test data. I will used this as main total "training" data in production & for cross_val stuff.
corpus = pd.read_csv('data/total_data.csv')

# Setting Up my X & y
X_raw = corpus.Comment
y_raw = corpus.Insult
    
# Setting Up My Different Pipelines. I tried out several different combos to see where to go next.
pipeline = Pipeline([
                     ('vect', TfidfVectorizer(binary=True, use_idf=True, stop_words='english')),
                     ('clf', LogisticRegression())
])

other_pipeline = Pipeline([
                     ('vect', CountVectorizer(binary=True)),
                     ('clf', LogisticRegression())
])


LSVC_pipeline = Pipeline([
                     ('vect',  TfidfVectorizer(binary=True, use_idf=True, stop_words='english')),
                     ('clf', LinearSVC())
])

Perceptron_pipeline = Pipeline([
                     ('vect',  TfidfVectorizer(binary=True, use_idf=True, stop_words='english')),
                     ('clf', Perceptron())
])


# # EVALUATION TIME!
# Here I am looking at **Accuracy**, **Precision**, & **Recall** ([learn more](http://en.wikipedia.org/wiki/Precision_and_recall)) and using the **cross_val_score** method which splits up my data, takes the **pipelines** I made, and then evaluates it over and over, in different combinations against itself to get a reliable score.
# 
# I want a good overall score but I would also like to improve my Recall as much as possible. Recall gives me a measurement of false negatives, meaning a score based on if I miss-classify something by saying a comment is NOT insulting when in fact it actually is. I am interested in a high score using this measurement to ensure [no hater is left behind](http://www.quickmeme.com/img/a3/a3e69566093ab74d0625a7c5892cade8c2adeecd86112356875ff2b05fcd020d.jpg). 

# In[229]:



# THE INITIAL 'STATNDARD' MODEL I STARTED WITH
accuracy_scores = cross_val_score(pipeline, X_raw, y_raw, cv=5)
precision_scores = cross_val_score(pipeline, X_raw, y_raw, cv=5, scoring='precision')
recall_scores = cross_val_score(pipeline, X_raw, y_raw, cv=5, scoring='recall')

print "***************OUR RESULTS***************"
print
print 

print "TfidfVect + Logistic Regression"
print "-------------------------------"
print "Accuracy:", accuracy_scores, "Mean:", np.mean(accuracy_scores)
print "Precision:", precision_scores, "Mean:", np.mean(precision_scores)
print "Recall:", recall_scores, "Mean:", np.mean(recall_scores)




# VS NORMAL COUNT VECTORIZER **NOTE** Out of all of the models without extra features, this performed the best for my goals.

other_accuracy_scores = cross_val_score(other_pipeline, X_raw, y_raw, cv=5)
other_precision_scores = cross_val_score(other_pipeline, X_raw, y_raw, cv=5, scoring='precision')
other_recall_scores = cross_val_score(other_pipeline, X_raw, y_raw, cv=5, scoring='recall')
print
print
print "CountVect + Logistic Regression"
print "-------------------------------"
print "Accuracy:", other_accuracy_scores, "Mean:", np.mean(other_accuracy_scores)
print "Precision:", other_precision_scores, "Mean:", np.mean(other_precision_scores)
print "Recall:", other_recall_scores, "Mean:", np.mean(other_recall_scores)




# VS NORMAL COUNT VECTORIZER

LSVC_accuracy_scores = cross_val_score(LSVC_pipeline, X_raw, y_raw, cv=5)
LSVC_precision_scores = cross_val_score(LSVC_pipeline, X_raw, y_raw, cv=5, scoring='precision')
LSVC_recall_scores = cross_val_score(LSVC_pipeline, X_raw, y_raw, cv=5, scoring='recall')
print
print
print "TfidfVect + LinearSVC"
print "-------------------------------"
print "Accuracy:", LSVC_accuracy_scores, "Mean:", np.mean(LSVC_accuracy_scores)
print "Precision:", LSVC_precision_scores, "Mean:", np.mean(LSVC_precision_scores)
print "Recall:", LSVC_recall_scores, "Mean:", np.mean(LSVC_recall_scores)





# VS NORMAL COUNT VECTORIZER

Perceptron_accuracy_scores = cross_val_score(Perceptron_pipeline, X_raw, y_raw, cv=5)
Perceptron_precision_scores = cross_val_score(Perceptron_pipeline, X_raw, y_raw, cv=5, scoring='precision')
Perceptron_recall_scores = cross_val_score(Perceptron_pipeline, X_raw, y_raw, cv=5, scoring='recall')
print
print
print "TfidfVect + Percptron"
print "-------------------------------"
print "Accuracy:", Perceptron_accuracy_scores, "Mean:", np.mean(Perceptron_accuracy_scores)
print "Precision:", Perceptron_precision_scores, "Mean:", np.mean(Perceptron_precision_scores)
print "Recall:", Perceptron_recall_scores, "Mean:", np.mean(Perceptron_recall_scores)


# 
# 
# #Now That I Have A Baseline, Time to Pick A Model And Build It Out Further. 
# 
# I ended it up picking **CountVectorizer + Logistic Regression** which you will see further down. In order to improve the score I started to play with extra features adding them manually and "hand making" them. These are almost meta data about each document that I was hoping would help give extra information to my classifier. They are things like How many "bad words" are used in a specific comment or the ratio of bad words used vs how many words total were in a comment. I even tried to look at things like if someone spoke in all CAPS or not to see if that could help predict if the comment is insulting or not.
# 

# In[230]:

# Setting up our actually train and test data sets
train_corpus = pd.read_csv('data/train.csv')
test_corpus = pd.read_csv('data/test_with_solutions.csv')

# Getting our list of "badwords"
badwords = set(pd.read_csv('data/my_badwords.csv').words)

# If you would like to see what some of these words look like, play with this:
print "Some Badwords We Will Check:", list(badwords)[0:9]
print 

# Setting Up my X & y
# NOTE: If you want to use all the data (including the test data), uncomment out the other X & y trains below.
X_train = train_corpus.Comment
y_train = train_corpus.Insult
# X_train = corpus.Comment
# y_train = corpus.Insult

X_test = test_corpus.Comment
y_test = test_corpus.Insult

# Just giving feedback so we know if we are using all of the data to train the model or not.
print "X_train's number of instances:",X_train.shape[0]
print

if X_train.shape[0] > train_corpus.Comment.shape[0]:
    print "*************************************************************************************************************"
    print "***JUST AN FYI, YOU'RE USING BOTH THE TRAINING & TESTING DATA FOR TRAINING. SHOULD BE FOR PRODUCTION ONLY.***"
    print "*************************************************************************************************************"
else:
    print "----------------------------------------------------------------------"
    print "You are using just the training data for training. Makes sense, right?"
    print "----------------------------------------------------------------------"
    


# #Adding In Extra Features.
# 
# Here I add in various extra features to help my classifier out. These are features that I had to manually create and add into my model so I didn't use cross_val_score and had to evaluate them manually. 
# 
# ###Features I added to improve the model are the following:
# * **badwords_count - **A count of bad words used in each comment.
# * **n_words - **A count of words used in each comment.
# * **allcaps - **A count of capital letters in each comment.
# * **allcaps_ratio - **A count of capital letters in each comment / the total words used in each comment.
# * **bad_ratio - **A count of bad words used in each comment / the total words used in each comment.
# * **exclamation - **A count of "!" used in each comment.
# * **addressing - **A count of "@" symbols used in each comment.
# * **spaces - **A count of spaces used in each comment.
# 

# In[231]:

# Since I was unhappy with aspects of my score, I added addtional features my classifier can use

# This is just a count of how many bad word
train_badwords_count = []
test_badwords_count = []

for el in X_train:
    tokens = el.split(' ')
    train_badwords_count.append(len([i for i in tokens if i.lower() in badwords]))
    
for el in X_test:
    tokens = el.split(' ')
    test_badwords_count.append(len([i for i in tokens if i.lower() in badwords]))
    
# **SHOUT OUT**
# I was messing with stuff from Andreas Mueller for these next features. Thanks man! :) 
# His Blog Post on this: http://blog.kaggle.com/2012/09/26/impermium-andreas-blog/ 
# **SHOUT OUT**
    
train_n_words = [len(c.split()) for c in X_train]
test_n_words = [len(c.split()) for c in X_test]

train_allcaps = [np.sum([w.isupper() for w in comment.split()]) for comment in X_train]
test_allcaps = [np.sum([w.isupper() for w in comment.split()]) for comment in X_test]

train_allcaps_ratio = np.array(train_allcaps) / np.array(train_n_words, dtype=np.float)
test_allcaps_ratio = np.array(test_allcaps) / np.array(test_n_words, dtype=np.float)

train_bad_ratio = np.array(train_badwords_count) / np.array(train_n_words, dtype=np.float)
test_bad_ratio = np.array(test_badwords_count) / np.array(test_n_words, dtype=np.float)

train_exclamation = [c.count("!") for c in X_train]
test_exclamation = [c.count("!") for c in X_test]

train_addressing = [c.count("@") for c in X_train]
test_addressing = [c.count("@") for c in X_test]

train_spaces = [c.count(" ") for c in X_train]
test_spaces = [c.count(" ") for c in X_test]


print "train_badwords count:", len(train_badwords_count), "test_badwords count:", len(test_badwords_count),
print "train_allcaps count:", len(train_allcaps), "test_allcaps count:", len(test_allcaps)


# #Time To Pick Our Model & Add In Our New Features.
# Now that we have our new features we need to pick a vectorizer, smash the new features inside of our vectorized old features, and evaluated everything with our classfier.

# In[232]:

# SIDE NOTE: TfidfVectorizer sucks for this type of problem. Well, you could use it and not use idf and it does get better but not too much...
# vect = TfidfVectorizer(binary=True, use_idf=True, stop_words='english')
vect = CountVectorizer(binary=True)
clf = LogisticRegression()


# ###*Side Note: This is the "smashing" process I was talking about.
# I'm basically fiting and transforming our features via our vectorizer and then useing [.reshape( )](http://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html) & [np.hstack( )](http://docs.scipy.org/doc/numpy/reference/generated/numpy.hstack.html) to manually add in our extra features.

# In[233]:

X_train_transform = vect.fit_transform(X_train)
X_test_transform = vect.transform(X_test)

train_reshaped_badwords = np.array(train_badwords_count).reshape((len(train_badwords_count),1))
test_reshaped_badwords = np.array(test_badwords_count).reshape((len(test_badwords_count),1))

train_reshaped_n_words = np.array(train_n_words).reshape((len(train_badwords_count),1))
test_reshaped_n_words = np.array(test_n_words).reshape((len(test_badwords_count),1))

train_reshaped_allcaps = np.array(train_allcaps).reshape((len(train_badwords_count),1))
test_reshaped_allcaps = np.array(test_allcaps).reshape((len(test_badwords_count),1))

train_reshaped_allcaps_ratio = np.array(train_allcaps_ratio).reshape((len(train_badwords_count),1))
test_reshaped_allcaps_ratio = np.array(test_allcaps_ratio).reshape((len(test_badwords_count),1))

train_reshaped_bad_ratio = np.array(train_bad_ratio).reshape((len(train_badwords_count),1))
test_reshaped_bad_ratio = np.array(test_bad_ratio).reshape((len(test_badwords_count),1))

train_reshaped_exclamation = np.array(train_exclamation).reshape((len(train_badwords_count),1))
test_reshaped_exclamation = np.array(test_exclamation).reshape((len(test_badwords_count),1))

train_reshaped_addressing = np.array(train_addressing).reshape((len(train_badwords_count),1))
test_reshaped_addressing = np.array(test_addressing).reshape((len(test_badwords_count),1))

train_reshaped_spaces = np.array(train_spaces).reshape((len(train_badwords_count),1))
test_reshaped_spaces = np.array(test_spaces).reshape((len(test_badwords_count),1))



X_train_transform = np.hstack((X_train_transform.todense(), train_reshaped_badwords))
X_test_transform = np.hstack((X_test_transform.todense(), test_reshaped_badwords))

X_train_transform = np.hstack((X_train_transform, train_reshaped_n_words))
X_test_transform = np.hstack((X_test_transform, test_reshaped_n_words))

X_train_transform = np.hstack((X_train_transform, train_reshaped_allcaps))
X_test_transform = np.hstack((X_test_transform, test_reshaped_allcaps))

X_train_transform = np.hstack((X_train_transform, train_reshaped_allcaps_ratio))
X_test_transform = np.hstack((X_test_transform, test_reshaped_allcaps_ratio))

X_train_transform = np.hstack((X_train_transform, train_reshaped_bad_ratio))
X_test_transform = np.hstack((X_test_transform, test_reshaped_bad_ratio))

X_train_transform = np.hstack((X_train_transform, train_reshaped_exclamation))
X_test_transform = np.hstack((X_test_transform, test_reshaped_exclamation))

X_train_transform = np.hstack((X_train_transform, train_reshaped_addressing))
X_test_transform = np.hstack((X_test_transform, test_reshaped_addressing))

X_train_transform = np.hstack((X_train_transform, train_reshaped_spaces))
X_test_transform = np.hstack((X_test_transform, test_reshaped_spaces))


# #Scoring Our New Model.
# Finally We get to see our new score. It seems to be doing a bit better but to start to look at recall and other measures we can set our predictions and then see how many items we missed and in what ways we missed them. Execute the final cells to see the scores.

# In[234]:

clf.fit(X_train_transform,y_train)
predictions = clf.predict(X_test_transform)
clf.score(X_test_transform, y_test)


# In[235]:

# Run this cell to get a print out of each of our wrong predcitions, and how they were predicted incorrectly.
wrong_predictions_number = 0
false_negatives = 0
false_postitives = 0
for i,p in enumerate(predictions):
    if p != y_test[i]:
        wrong_predictions_number += 1
        if p > y_test[i]:
            false_postitives += 1
            print
            print "++FALSE POSITIVE++"
        if p < y_test[i]:
            false_negatives += 1
            print
            print "--FALSE NEGATIVE--"
        print "predicted:", p, "actual", y_test[i]


# In[236]:

# Execute this cell to see a final scoring of our new model
print "total number of instances:", len(predictions), 
print "| total number wrong:", wrong_predictions_number, 
print "| total false negatives:", false_negatives, 
print "| total false positives:", false_postitives
print
print
print "            THE FINAL STATS"
print "----------------------------------------"
print "Percent Right:", 100 - (float(wrong_predictions_number) / float(len(predictions))) * 100,"%"
print "Percent Wrong:", (float(wrong_predictions_number) / float(len(predictions))) * 100,"%"
print "Percent False Negatives:", (float(false_negatives) / float(len(predictions))) * 100,"%"
print "Percent False Positives:", (float(false_postitives) / float(len(predictions))) * 100,"%"


# In[147]:

# Uncomment this and execute to get a list of insults to look through for more potential patterns
# corpus.Comment[corpus.Insult == 1].to_csv("insults.csv")


# In[ ]:



