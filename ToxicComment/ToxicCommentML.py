import pandas as pd 
import random as rand 
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

# Import data
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

combined_text = train_df.append(test_df, ignore_index=True)
combined_comment = combined_text['comment_text']
vectorizer = TfidfVectorizer(lowercase=True, stop_words='english')
tfidf_matrix = vectorizer.fit(combined_comment)

train_features_X = tfidf_matrix.transform(train_df['comment_text'])
test_features_X = tfidf_matrix.transform(test_df['comment_text'])

# Classifier
clf = RandomForestClassifier()
y = train_df.iloc[:, 2:8]
clf.fit(train_features_X, y)

labels = clf.predict(test_features_X)
submission = pd.DataFrame(labels, columns=['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'])
submission['id'] = test_df['id']

cols = submission.columns.tolist()
cols = ['id', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
submission = submission[cols]

submission.to_csv('first_submission1.csv', index=None)