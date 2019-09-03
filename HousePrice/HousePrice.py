import pandas as pd 
import random as rand 
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# Drop Nan and str columns
clean_train_df = train_df.dropna(axis='columns')
clean_train_df = clean_train_df.select_dtypes(exclude=['object'])
clean_test_df = test_df.dropna(axis='columns')
clean_test_df = clean_test_df.select_dtypes(exclude=['object'])

# Make columns the same
clean_train_df = clean_train_df.loc[:,clean_test_df.columns]
clean_train_df['SalePrice'] = train_df['SalePrice']

y = clean_train_df['SalePrice']
x = clean_train_df[clean_train_df.columns[:-1]]

# Train regressor
regr = RandomForestRegressor()
regr.fit(x, y)

# Predict
test_predictions = regr.predict(clean_test_df)
test_predictions = pd.DataFrame(test_predictions)
test_predictions.index.name ='Id'
test_predictions.columns = ['SalePrice']
test_predictions.index += 1461
test_predictions.to_csv('final_submission3.csv', index=True)