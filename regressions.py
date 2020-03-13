import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

def time_to_min(string):
    if string is not '-':
        time_segments = string.split(':')
        return int(time_segments[0])*60 + int(time_segments[1]) + np.true_divide(int(time_segments[2]), 60)
    else:
        return -1

def gender_to_numeric(value):
    if value == 'M':
        return 0
    else:
        return 1

path = ''
filename = ''
df = pd.read_csv(path + filename)

df = df.drop(index=8, columns=['Bib', 'Name', 'City', 'State', 'Country', 'Citizen'], axis=1)
df = df.drop(df.columns[2], axis=1)

df['M/F'] = df['M/F'].apply(lambda x: gender_to_numeric(x))

# remove damaged data
df = df[(~(df['5K'] == '-')) &(~(df['10K'] == '-'))&(~(df['15K'] == '-'))&(~(df['20K'] == '-'))&(~(df['25K'] == '-')) &(~(df['30K'] == '-')) &(~(df['35K'] == '-')) &(~(df['40K'] == '-'))&(~(df['Half'] == '-'))]

# unify data
df['Half'] = df.Half.apply(lambda x: time_to_min(x))
df['Full'] = df['Official Time'].apply(lambda x: time_to_min(x))
df['split_ratio'] = (df['Full'] - df['Half'])/(df['Half'])

df['5K_mins']  = df['5K'].apply(lambda x: time_to_min(x))
df['10K_mins'] = df['10K'].apply(lambda x: time_to_min(x))
df['10K_mins'] = (df['10K_mins'] - df['5K_mins'])

df['15K_mins'] = df['15K'].apply(lambda x: time_to_min(x))
df['15K_mins'] = (df['15K_mins'] - df['10K_mins'] -  df['5K_mins'])

df['20K_mins'] = df['20K'].apply(lambda x: time_to_min(x))
df['20K_mins'] = (df['20K_mins'] -  df['15K_mins'] - df['10K_mins'] -  df['5K_mins'])

df['25K_mins'] = df['25K'].apply(lambda x: time_to_min(x))
df['25K_mins'] = (df['25K_mins'] - df['20K_mins'] -  df['15K_mins'] - df['10K_mins']
                  - df['5K_mins'])

df['30K_mins'] = df['30K'].apply(lambda x: time_to_min(x))
df['30K_mins'] = (df['30K_mins'] -df['25K_mins'] - df['20K_mins'] - df['15K_mins'] 
                  - df['10K_mins'] - df['5K_mins'])

df['35K_mins'] = df['35K'].apply(lambda x: time_to_min(x))
df['35K_mins'] = (df['35K_mins'] -df['30K_mins'] - df['25K_mins'] - df['20K_mins'] 
                    - df['15K_mins'] - df['10K_mins'] - df['5K_mins'])

df['40K_mins'] = df['40K'].apply(lambda x: time_to_min(x))
df['40K_mins'] = (df['40K_mins'] -  df['35K_mins'] - df['30K_mins'] -df['25K_mins'] 
                    - df['20K_mins'] -  df['15K_mins'] - df['10K_mins'] - df['5K_mins'])


columns = ['40K_mins', '35K_mins', '30K_mins', '25K_mins', '20K_mins','15K_mins','10K_mins','5K_mins']
df['avgOf5K'] = df[columns].mean(axis = 1)
df['stdev'] = df[columns].std(axis = 1)

prediction_df = df[['Age', 'M/F', 'Half', 'Full', 'split_ratio',
                    '40K_mins', '35K_mins', '30K_mins', '25K_mins', '5K_mins','10K_mins','15K_mins','20K_mins',
                    'avgOf5K', 'stdev']]

traindf, testdf = train_test_split(prediction_df, test_size = 0.2)

X_train = traindf[['Age', 'Half', '5K_mins', '10K_mins', '15K_mins', '20K_mins', 'stdev']]

y_train = traindf['Full']

X_test = testdf[['Age', 'Half', '5K_mins', '10K_mins', '15K_mins', '20K_mins', 'stdev']]
y_test = testdf['Full']

# Train
LR = LinearRegression()
LR.fit(X_train, y_train)

# Test
LR_train_pred = LR.predict(X_train) # predict with testing data
LR_test_pred  = LR.predict(X_test)

print('Linear Regression:')
print('MSE train: %.2f, test: %.2f' % (
                mean_squared_error(y_train, LR_train_pred),
                mean_squared_error(y_test, LR_test_pred)))
print('R^2 train: %.2f, test: %.2f' % (
                r2_score(y_train, LR_train_pred),
                r2_score(y_test, LR_test_pred)))


tree = DecisionTreeRegressor(max_depth=11)
tree.fit(X_train, y_train)

DTR_train_pred = tree.predict(X_train)
DTR_test_pred = tree.predict(X_test)

print('Decision Tree Regressor:')
print('MSE train: %.2f, test: %.2f' % (
                mean_squared_error(y_train, DTR_train_pred),
                mean_squared_error(y_test, DTR_test_pred)))
print('R^2 train: %.2f, test: %.2f' % (
                r2_score(y_train, DTR_train_pred),
                r2_score(y_test, DTR_test_pred)))

XGB = XGBRegressor(learning_rate=0.01, n_estimators = 2000)

XGB.fit(X_train,y_train)

XGB_train_pred = XGB.predict(X_train)
XGB_test_pred = XGB.predict(X_test)

print('XGB Regressor')
print('MSE train: %.2f, test: %.2f' % (
                mean_squared_error(y_train, XGB_train_pred),
                mean_squared_error(y_test, XGB_test_pred)))

print('R^2 train: %.2f, test: %.2f' % (
                r2_score(y_train, XGB_train_pred),
                r2_score(y_test, XGB_test_pred)))

print('--------\nTrain by 5Ks:')

X_train_1 = traindf[['Age', '5K_mins']]
y_train_1 = traindf[['10K_mins']]

X_test_1 = testdf[['Age', '5K_mins']]
y_test_1 = testdf[['10K_mins']]

model = LinearRegression()
model.fit(X_train_1, y_train_1)

pred_1 = model.predict(X_test_1)

print('Iteration 1:')
print('MSE: %.2f' % mean_squared_error(y_test_1, pred_1))
print('R^2: %.2f' % r2_score(y_test_1, pred_1))

X_train_2 = traindf[['Age', '5K_mins', '10K_mins']]
y_train_2 = traindf[['15K_mins']]

X_test_2 = testdf[['Age', '5K_mins', '10K_mins']]
y_test_2 = testdf[['15K_mins']]

model.fit(X_train_2, y_train_2)

pred_2 = model.predict(X_test_2)

print('Iteration 2:')
print('MSE: %.2f' % mean_squared_error(y_test_2, pred_2))
print('R^2: %.2f' % r2_score(y_test_2, pred_2))

X_train_3 = traindf[['Age', '5K_mins', '10K_mins', '15K_mins']]
y_train_3 = traindf[['20K_mins']]

X_test_3 = testdf[['Age', '5K_mins', '10K_mins', '15K_mins']]
y_test_3 = testdf[['20K_mins']]

model.fit(X_train_3, y_train_3)

pred_3 = model.predict(X_test_3)

print('Iteration 3:')
print('MSE: %.2f' % mean_squared_error(y_test_3, pred_3))
print('R^2: %.2f' % r2_score(y_test_3, pred_3))


X_train_4 = traindf[['Age', '5K_mins', '10K_mins', '15K_mins', '20K_mins']]
y_train_4 = traindf[['25K_mins']]

X_test_4 = testdf[['Age', '5K_mins', '10K_mins', '15K_mins', '20K_mins']]
y_test_4 = testdf[['25K_mins']]

model.fit(X_train_4, y_train_4)

pred_4 = model.predict(X_test_4)

print('Iteration 4:')
print('MSE: %.2f' % mean_squared_error(y_test_4, pred_4))
print('R^2: %.2f' % r2_score(y_test_4, pred_4))


X_train_5 = traindf[['Age', '5K_mins', '10K_mins', '15K_mins', '20K_mins', '25K_mins']]
y_train_5 = traindf[['30K_mins']]

X_test_5 = testdf[['Age', '5K_mins', '10K_mins', '15K_mins', '20K_mins', '25K_mins']]
y_test_5 = testdf[['30K_mins']]

model.fit(X_train_5, y_train_5)

pred_5 = model.predict(X_test_5)

print('Iteration 5:')
print('MSE: %.2f' % mean_squared_error(y_test_5, pred_5))
print('R^2: %.2f' % r2_score(y_test_5, pred_5))


X_train_6 = traindf[['Age', '5K_mins', '10K_mins', '15K_mins', '20K_mins', '25K_mins', '30K_mins']]
y_train_6 = traindf[['35K_mins']]

X_test_6 = testdf[['Age', '5K_mins', '10K_mins', '15K_mins', '20K_mins', '25K_mins', '30K_mins']]
y_test_6 = testdf[['35K_mins']]

model.fit(X_train_6, y_train_6)

pred_6 = model.predict(X_test_6)

print('Iteration 6:')
print('MSE: %.2f' % mean_squared_error(y_test_6, pred_6))
print('R^2: %.2f' % r2_score(y_test_6, pred_6))


X_train_7 = traindf[['Age', '5K_mins', '10K_mins', '15K_mins', '20K_mins', '25K_mins', '30K_mins', '35K_mins']]
y_train_7 = traindf[['40K_mins']]

X_test_7 = testdf[['Age', '5K_mins', '10K_mins', '15K_mins', '20K_mins', '25K_mins', '30K_mins', '35K_mins']]
y_test_7 = testdf[['40K_mins']]

model.fit(X_train_7, y_train_7)

pred_7 = model.predict(X_test_7)

print('Iteration 7:')
print('MSE: %.2f' % mean_squared_error(y_test_7, pred_7))
print('R^2: %.2f' % r2_score(y_test_7, pred_7))


X_train_8 = traindf[['Age', '5K_mins', '10K_mins', '15K_mins', '20K_mins', '25K_mins', '30K_mins', '35K_mins', '40K_mins']]
y_train_8 = traindf[['Full']]

X_test_8 = testdf[['Age', '5K_mins', '10K_mins', '15K_mins', '20K_mins', '25K_mins', '30K_mins', '35K_mins', '40K_mins']]
y_test_8 = testdf[['Full']]

model.fit(X_train_8, y_train_8)

pred_8 = model.predict(X_test_8)

print('Iteration 8:')
print('MSE: %.2f' % mean_squared_error(y_test_8, pred_8))
print('R^2: %.2f' % r2_score(y_test_8, pred_8))
