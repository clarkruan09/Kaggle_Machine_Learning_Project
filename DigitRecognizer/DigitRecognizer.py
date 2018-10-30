import pandas as pd 
import random as rand 
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# Import data
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# Try show image
vector = train_df.loc[30000, :][1:]
img = np.array(vector).reshape(28,28)
plt.imshow(img)
plt.show()

# Classifier
clf = RandomForestClassifier()
X, y = train_df.iloc[:, 1:] ,train_df['label']
clf.fit(X, y)

# Predict
test_predictions = clf.predict(test_df)
test_predictions = pd.DataFrame(test_predictions)
test_predictions.index+=1
test_predictions.index.name ='ImageId'
test_predictions.columns = ['Label']
test_predictions.to_csv('final_submission.csv', index=True)