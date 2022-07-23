# imports

import nltk
import string
import unicodedata
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import regex as re

from nltk.corpus import stopwords
from sklearn import model_selection, svm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

nltk.download('stopwords')

# Example importing the CSV here
df = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/NLP-project-tutorial/main/url_spam.csv')
print(f'Spam Balance:\n {df["is_spam"].value_counts()}')


# Remove non-alpha (except '.' and '/'), numbers, tags, lonely letters
clean = []
for i in range(len(df.url)) :
    desc = df['url'][i]

    # remove non-aplha
    desc = re.sub('[^a-zA-Z]', ' ', desc)

    # remove numbers
    desc = re.sub('[0-9]', '', desc)

    # remove tags
    desc = re.sub('&lt;/?.*?&gt;', ' &lt;&gt; ', desc)

    # remove some known unnecesary words (http, https, www, com)
    desc = re.sub('https|http|www|html|\scom\s|\sorg\s|\sus\s', '', desc)

    # remove lonely letters
    for i in range(5) :
        desc = re.sub('\s[a-zA-Z]\s', ' ', desc)

    clean.append(desc)

df['url'] = clean


# Drop NAs & duplicates
df = df.dropna().drop_duplicates()
df = df.reset_index(inplace = False)[['url','is_spam']]
df.shape

# Sprip spaces & to lowercase
df['url'] = [entry.lower().strip() for entry in df['url']]

# is_spam True --> 1, False --> 0
df['is_spam'] = df['is_spam'].apply(lambda x: 1 if x == True else 0)


# Remove stop words:

stop = stopwords.words('english')
def remove_stopwords(Message):
  if Message is not None :
    words = Message.strip().split()
    words_filtered = []
    for word in words :
      if word not in stop :
        words_filtered.append(word)
    result = ' '.join(words_filtered)
  else :
    result = None
  return result

df['url'] = df['url'].apply(remove_stopwords)


# Split train test models
X = df['url']
y = df['is_spam']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=13)


# Vectorizador
vec = CountVectorizer(stop_words='english')
X_train = vec.fit_transform(X_train).toarray()
X_test = vec.transform(X_test).toarray()


# Generate model
nb = MultinomialNB()
nb.fit(X_train, y_train)

# Train Score
print(f'Train Score: {nb.score(X_train, y_train)}')

# Test Score
print(f'Test Score: {nb.score(X_test, y_test)}')