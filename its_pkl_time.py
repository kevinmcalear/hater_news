# (Re)Training the model on application startup is wasteful.
# Let's train a model and save it to disk.
# The pickle module implements a fundamental, but powerful algorithm
# for serializing and de-serializing a Python object structure.
# Pickling is the process whereby a Python object hierarchy is
# converted into a byte stream, and unpickling is the inverse
# operation, whereby a byte stream is converted back into an object
# hierarchy. Pickling (and unpickling) is alternatively known as
# serialization, marshalling, or flattening, however,
# to avoid confusion, the terms used here are pickling and unpickling.
# sklearn provides a modified pickler that works better with large NumPy arrays.
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

print 'Training model...'
df = pd.read_csv('lyrics.csv')
y = df['Artist']
X = df['Lyrics']

pipeline = Pipeline([
    ('vect', TfidfVectorizer()),
    ('clf', LogisticRegression())
])
pipeline.fit(X, y)

# TODO import joblib from sklearn.externals
from sklearn.externals import joblib
# TODO dump the fitted pipeline to a pickle called lyrics_pipeline.pkl
joblib.dump(pipeline, 'lyrics_pipeline.pkl')
