import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Read in testing and training data into two dataframes
test_df=pd.read_csv("test.csv")
train_df=pd.read_csv("train.csv")

train_df['author'].value_counts()

# Split into features and labels
X_train = train_df['text']
y_train = train_df['author']
X_test = test_df['text']


# We need to transform our text data into vectors so that we can run it though a machine learning model
vectorizer = CountVectorizer(stop_words='english')
corpus = pd.concat([train_df['text'], test_df['text']])
vectorizer.fit(corpus)

X_train = vectorizer.transform(X_train)
X_test = vectorizer.transform(X_test)

classifier = MultinomialNB()
classifier.fit(X_train, y_train)

y_pred_proba = classifier.predict_proba(X_test)

submission = pd.DataFrame(y_pred_proba, columns=["EAP","HPL","MWS"])
submission['id'] = test_df['id']

submission = submission[["id","EAP","HPL","MWS"]]
submission.to_csv('submission.csv', index=None)