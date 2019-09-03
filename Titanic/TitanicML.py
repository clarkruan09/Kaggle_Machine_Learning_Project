import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt

# Load scikit's random forest classifier library
from sklearn.ensemble import RandomForestClassifier

# Read in training and testing csvs
train_df = pd.read_csv(open('train.csv'))
test_df = pd.read_csv(open('train.csv'))

# Clean the train dataset: dropping name, ticket, cabin, ID, and embarked columns
train_df.drop(['Name', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True) 
# One-hot encoding Gender
train_df['m'] = (train_df['Sex'] == 'male')
train_df['f'] = (train_df['Sex'] == 'female')
train_df.drop(['Sex'], axis=1, inplace=True)
# Replace Nan values in age column with average age
train_mean_age = train_df['Age'].mean()
train_df.fillna(value=train_mean_age, inplace=True);

# Clean the test dataset in the same way as the train dataset: dropping name, ticket, cabin, ID, and embarked columns
test_df.drop(['Name', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True) 
# One-hot encoding Gender
test_df['m'] = (test_df['Sex'] == 'male')
test_df['f'] = (test_df['Sex'] == 'female')
test_df.drop(['Sex'], axis=1, inplace=True)
# Replace Nan values in age column with average age
test_mean_age = test_df['Age'].mean()
test_df.fillna(value=test_mean_age, inplace=True);
#Fill NaN value in fare column with mean fare in only the test dataset as it has some missing values
test_df['Fare'].fillna(test_df['Fare'].mean(), inplace=True)
# Drop the PassengerId column from the training dataset as we will not need it for training
train_df.drop(['PassengerId'], axis=1, inplace=True)


#Classifier
clf = RandomForestClassifier(n_jobs=2, random_state=0)
y = train_df['Survived']
x = train_df.iloc[:,1:]
clf.fit(x, y);

#Predict
survive = clf.predict(test_df.iloc[:, 2:])
final = list(zip(test_df['PassengerId'], survive))
final_prediction = pd.DataFrame(final, columns=['PassengerId', 'Survived'])
final_prediction.to_csv('final_prediction3.csv', index=False)

